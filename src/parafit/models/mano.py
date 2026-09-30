"""ManoModel -- MANO hand under the parafit Model interface. [extra: mano]

Parameters ``theta = [pose(48) | trans(3)] = 51`` DOF; the shape ``betas`` (10)
is a fixed side input (predicted by a compact head upstream, not optimized in
the fit). The forward is manotorch's ``ManoLayer`` (``flat_hand_mean`` / ``center_idx`` configurable, see ManoModel)
followed by ``mano_to_openpose`` (16 regressed joints + 5 fingertips -> 21
OpenPose joints). The analytic landmark Jacobian is the cross-product
articulated Jacobian with the SO(3) right-Jacobian correction, ported from the
UA-Fit solver (``mvhpe/POEM-v2/lib/models/dovf/analytic_fitter.py::mano_kinematic_jac``);
it is validated against finite differences in the tests. MANO weights are not
shipped (license): pass the path to your ``mano_v1_2`` assets.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from parafit.core.lie import so3_right_jacobian
from parafit.core.types import ParamSpec, State
from parafit.models.base import Model
from parafit.registry import register_model

# Fingertip vertex ids (MANO KPID -> vertex), OpenPose reorder, and the distal
# MANO joint each fingertip rides on (thumb, index, middle, ring, pinky).
_TIP_VERTS = [744, 320, 443, 555, 672]
_OPENPOSE_PERM = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
_TIP_DISTAL = [15, 3, 6, 12, 9]
# smplx / HaMeR / HaWoR fingertip vertices (their MANO wrappers' `vertex_ids['mano']`)
_TIP_VERTS_SMPLX = [744, 320, 443, 554, 671]
_OP_TIPS = [4, 8, 12, 16, 20]
# manotorch's own fingertip vertices (ManoLayer.skinning_layer); its `center_idx` joint is taken from its joints, so a
# fingertip centre is one of these
_MT_TIPS = {"right": [745, 317, 444, 556, 673], "left": [745, 317, 445, 556, 673]}


@register_model("mano", extra="mano")
def _build_mano(*args, **kwargs) -> "ManoModel":
    return ManoModel(*args, **kwargs)


def _hat(v: Tensor) -> Tensor:
    z = torch.zeros(v.shape[:-1], dtype=v.dtype, device=v.device)
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def _ancestor_mask(parents, n: int = 16) -> Tensor:
    """A[k, j] = 1 if joint j is on the kinematic path root..k (inclusive)."""
    A = torch.zeros(n, n)
    for k in range(n):
        j = k
        while True:
            A[k, j] = 1.0
            p = int(parents[j])
            if j == 0 or p < 0 or p >= n:
                break
            j = p
        A[k, 0] = 1.0
    return A


def _mano_to_openpose(J_regressor: Tensor, verts: Tensor) -> Tensor:
    """MANO verts (B,778,3) -> 21 OpenPose joints (B,21,3)."""
    joints = torch.matmul(J_regressor, verts)                    # (B,16,3)
    tips = verts[:, _TIP_VERTS]                                  # (B,5,3)
    stacked = torch.cat([joints, tips], dim=1)                   # (B,21,3)
    return stacked[:, _OPENPOSE_PERM]


class ManoModel(Model):
    def __init__(self, mano_assets_root: str, betas: Optional[Tensor] = None,
                 num_joints: int = 21, center_idx: Optional[int] = 0, side: str = "right",
                 flat_hand_mean: bool = True, joints: str = "regressed", fix_left_shapedirs: bool = False):
        """``flat_hand_mean`` and ``center_idx`` are conventions of whatever
        produced the parameters, not preferences. Mismatching them is a silent
        ~100 mm error, so they are exposed rather than hardcoded.

        ``center_idx=0`` centres the output on the wrist before adding ``trans``;
        ``center_idx=None`` leaves the MANO template offset in, so the wrist sits
        ~9.7 cm from ``trans``. ``flat_hand_mean`` selects whether zero pose means
        a flat hand or the MANO mean pose.

        Measured settings per producer (all verified numerically, see
        ``tests/test_mano_conventions.py``):

        ========================  ================  ==========  ==============
        producer                  flat_hand_mean    center_idx  wrong-setting
        ========================  ================  ==========  ==============
        UA-Fit (parafit default)  True              0           --
        ARCTIC GT params          **False**         None        100 mm
        WiLoR / HaWoR             **True**          None        30.7 mm
        ========================  ================  ==========  ==============

        ARCTIC and the WiLoR/HaWoR family differ, which is not obvious and is
        worth stating plainly. ARCTIC stores axis-angle parameters consumed by an
        axis-angle MANO, which DOES add the mean pose. WiLoR and HaWoR predict
        rotation matrices consumed by ``smplx.MANOLayer``, which does NOT, because
        the mean pose is only ever applied in axis-angle space.

        **``smplx.MANOLayer.flat_hand_mean`` reports ``False`` while behaving as
        ``True``.** It even carries a nonzero ``pose_mean`` (sum ~11.7). Reading
        that flag and configuring downstream code from it is a silent 30.7 mm
        error. This also bites when consuming HaWoR's *stored* output, which is
        axis-angle (converted via ``rotation_matrix_to_angle_axis``) but must
        still be evaluated with ``flat_hand_mean=True`` to match the prediction.

        ``joints``: "regressed" = J_regressor applied to the POSED vertices (default, UA-Fit convention) or "kinematic" =
        the skeleton's joint centres (the 16 transform origins) + smplx fingertip vertices, the convention of HaMeR / WiLoR /
        HaWoR outputs and of their GT exports (they differ by up to ~8 mm at the knuckles for the same pose).

        fix_left_shapedirs: the MANO_LEFT shape directions have the wrong sign in their x component (the left template is the
        exact x-mirror of the right one, the left shapedirs are NOT): with nonzero betas the left hand is distorted (23 mm max at
        beta0 = 2). True negates shapedirs[:, 0, :] for side="left" (HaMeR / HaWoR "fix_shapedirs"); off by default for
        reproducibility of existing fits.

        All analytic Jacobians (rigid rows, ``tip_corrective`` exact rows, ``_vertex_jacobian``) are taken at the FULL
        pose ``theta + hands_mean`` (the rotations the layer applies) and are centred exactly like the forward.
        """
        if joints not in ("regressed", "kinematic"): raise ValueError(f"joints must be 'regressed' or 'kinematic', got {joints!r}")
        self.joints = joints
        self._tips = _TIP_VERTS_SMPLX if joints == "kinematic" else _TIP_VERTS
        from manotorch.manolayer import ManoLayer  # raises ImportError -> parafit[mano]

        self.mano_layer = ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_assets_root=mano_assets_root,
            center_idx=center_idx,
            flat_hand_mean=flat_hand_mean,
            side=side,
        )
        if fix_left_shapedirs and side == "left":
            self.mano_layer.th_shapedirs[:, 0, :] *= -1.0          # MANO_LEFT sign bug (see docstring); every forward / Jacobian reads this buffer
        self.fix_left_shapedirs = bool(fix_left_shapedirs and side == "left")
        self.J_regressor = self.mano_layer.th_J_regressor       # (16,778)
        self.num_joints = num_joints
        self.center_idx = center_idx
        self.flat_hand_mean = flat_hand_mean
        self.betas = betas                                       # (B,10) or None -> zeros
        self._anc = _ancestor_mask(self.mano_layer.kintree_parents)  # (16,16)
        self.tip_corrective = False   # True: EXACT analytic landmark Jacobian (pose correctives + skinning blend; joints are regressed from the posed vertices, so every landmark carries these terms)
        self._exact_idx = None
        self.param_spec = ParamSpec({"pose": (0, 48), "trans": (48, 3)})

    def _full_theta(self, params: Tensor) -> Tensor:
        """(B,16,3) the axis-angles the layer rotates by: pose + the MANO mean pose unless flat_hand_mean (a constant
        offset, so Jacobians w.r.t. the parameters are Jacobians w.r.t. this)"""
        B = params.shape[0]
        mean = self.mano_layer.th_hands_mean.to(params).reshape(1, 45)
        return torch.cat([params[:, :3], params[:, 3:48] + mean], 1).reshape(B, 16, 3)

    def _betas_for(self, B: int, ref: Tensor) -> Tensor:
        if self.betas is None:
            return torch.zeros(B, 10, dtype=ref.dtype, device=ref.device)
        b = self.betas.to(ref)
        return b.expand(B, 10) if b.dim() == 2 and b.shape[0] == 1 else b

    def forward(self, params: Tensor) -> State:
        B = params.shape[0]
        pose = params[:, :48]
        trans = params[:, 48:51]
        betas = self._betas_for(B, params)
        out = self.mano_layer(pose, betas)
        verts = out.verts                                        # (B,778,3), centered at center_idx
        if self.joints == "kinematic":
            jc = out.joints.clone(); jc[:, _OP_TIPS] = verts[:, self._tips]                # manotorch joints: transform origins, OpenPose order
            jc = jc[:, : self.num_joints]
        else:
            jc = _mano_to_openpose(self.J_regressor.to(params), verts)[:, : self.num_joints]
        landmarks = jc + trans.unsqueeze(1)
        return State(landmarks=landmarks, verts=verts + trans.unsqueeze(1),
                     transforms=out.transforms_abs, batch_size=[B])

    def _vertex_jacobian(self, params: Tensor, state: State, verts_idx, centered: bool = True) -> Tensor:
        """(B, V, 3, 48) EXACT analytic Jacobian of selected MANO vertices: pose blend shapes + the full skinning blend.
        centered=True (default) is the Jacobian of the vertices the forward returns (the layer subtracts its
        `center_idx` joint, see `_center_jacobian`); centered=False is the uncentred skinned vertex.
            v = sum_k w_k [ p_k + R_k (v_rest + Pc(theta) - J_k) ],   Pc = posedirs @ vec(R_1..15 - I)
            dv/dtheta_j = a_j x (sum_k w_k anc_kj q_k - (sum_k w_k anc_kj) p_j) + (sum_k w_k R_k) dPc/dtheta_j
        a_j are the world DOF axes (as in the rigid rows), q_k the k-th bone's contribution, and
        d vec(R_j)/d delta_m = R_j [ (J_r(theta_j))_{:,m} ]_x for the right-perturbation convention the solver uses."""
        from parafit.core.lie import so3_exp
        B = params.shape[0]; ml = self.mano_layer; dtp = params.dtype; dev = params.device
        T = state.transforms; Rg = T[:, :, :3, :3]; pg = T[:, :, :3, 3]
        theta = self._full_theta(params); Jr = so3_right_jacobian(theta); axes = Rg @ Jr
        betas = self._betas_for(B, params)
        shapedirs = ml.th_shapedirs.to(params); v_shaped = ml.th_v_template.to(params) + torch.einsum("vdk,bk->bvd", shapedirs, betas)
        J_rest = torch.matmul(self.J_regressor.to(params), v_shaped)                                      # (B,16,3)
        w = ml.th_weights.to(params)[verts_idx]                                                           # (V,16)
        posed = ml.th_posedirs.to(params)[verts_idx]                                                      # (V,3,135)
        R_loc = so3_exp(theta.reshape(-1, 3)).reshape(B, 16, 3, 3)
        I3 = torch.eye(3, dtype=dtp, device=dev)
        feat = (R_loc[:, 1:] - I3).reshape(B, 135)
        v_posed = v_shaped[:, verts_idx] + torch.einsum("vdk,bk->bvd", posed, feat)                       # (B,V,3)
        q = pg[:, None] + torch.einsum("bkij,bvkj->bvki", Rg, v_posed[:, :, None, :] - J_rest[:, None, :, :])   # (B,V,16,3)
        anc = self._anc.to(params)
        m = torch.einsum("vk,kj->vj", w, anc)                                                             # (V,16)
        sq = torch.einsum("vk,kj,bvki->bvji", w, anc, q)                                                  # (B,V,16,3)
        d = sq - m[None, :, :, None] * pg[:, None]                                                        # (B,V,16,3)
        V = len(verts_idx)
        cr = torch.cross(axes[:, None].expand(B, V, 16, 3, 3), d[..., None].expand(B, V, 16, 3, 3), dim=3)
        dP = cr.permute(0, 1, 3, 2, 4).reshape(B, V, 3, 48)
        # corrective term (vectorised over the 15 non-root joints)
        hatJ = _hat(Jr[:, 1:].permute(0, 1, 3, 2).reshape(B, 15 * 3, 3)).reshape(B, 15, 3, 3, 3)          # [.,j,m] = hat of the m-th axis
        dR = torch.einsum("bjik,bjmkl->bjmil", R_loc[:, 1:], hatJ)                                        # (B,15,3,3,3): d R_j / d delta_{j,m}
        dfeat = params.new_zeros(B, 15, 9, 16, 3)
        val = dR.reshape(B, 15, 3, 9).permute(0, 1, 3, 2)                                        # (B,15,9,3): d feat-block(j-1) / d theta_j
        for j in range(15): dfeat[:, j, :, j + 1, :] = val[:, j]                                 # block j-1 of feat depends only on theta_{j+1}
        dfeat = dfeat.reshape(B, 135, 48)
        dPc = torch.einsum("vdk,bkp->bvdp", posed, dfeat)
        Rw = torch.einsum("vk,bkij->bvij", w, Rg)
        dV = dP + torch.einsum("bvij,bvjp->bvip", Rw, dPc)
        return dV - self._center_jacobian(params, state)[:, None] if centered else dV

    def _skeleton_jacobian(self, params: Tensor, state: State) -> Tensor:
        """(B,16,3,48) exact Jacobian of the 16 skeleton joint centres (transform origins, MANO order): rigid rows
        a_j x (p_k - p_j) over the ancestors j of k"""
        B = params.shape[0]
        T = state.transforms; Rg = T[:, :, :3, :3]; pg = T[:, :, :3, 3]
        axes = Rg @ so3_right_jacobian(self._full_theta(params))
        diff = pg.unsqueeze(2) - pg.unsqueeze(1)                                                          # (B,16,16,3)
        cr = torch.cross(axes.unsqueeze(1).expand(B, 16, 16, 3, 3), diff.unsqueeze(-1).expand(B, 16, 16, 3, 3), dim=3)
        cr = cr * self._anc.to(params).view(1, 16, 16, 1, 1)
        return cr.permute(0, 1, 3, 2, 4).reshape(B, 16, 3, 48)

    def _center_jacobian(self, params: Tensor, state: State) -> Tensor:
        """(B,3,48) Jacobian of the point the layer centres on: manotorch's joint `center_idx` in its OpenPose-ordered
        joints = a skeleton joint centre (rigid row; zero for the root, the default) or one of its fingertip vertices"""
        if self.center_idx is None:                                                                      # the layer does not centre
            return params.new_zeros(params.shape[0], 3, 48)
        k = _OPENPOSE_PERM[self.center_idx]
        if k == 0:                                                                                       # the root: fixed
            return params.new_zeros(params.shape[0], 3, 48)
        if k < 16:
            return self._skeleton_jacobian(params, state)[:, k]
        vid = _MT_TIPS[self.mano_layer.side][k - 16]
        return self._vertex_jacobian(params, state, [vid], centered=False)[:, 0]

    def _tip_jacobian(self, params: Tensor, state: State) -> Tensor:
        """Exact (B,5,3,48) Jacobian of the fingertip VERTICES: they carry the pose blend shapes and are skinned by a blend
        of joint transforms, which the rigid-ride rows ignore (the 16 joint centres are exact there). Closed form:
            v = sum_k w_k [ p_k + R_k (v_rest + Pc(theta) - J_k) ],  Pc = posedirs @ vec(R_1..15 - I)
            dv/dtheta_j = a_j x (sum_k w_k anc_kj q_k - (sum_k w_k anc_kj) p_j) + (sum_k w_k R_k) dPc/dtheta_j
        with q_k the k-th bone's contribution and a_j the world DOF axes (the same `axes` the rigid rows use)."""
        from parafit.core.lie import so3_exp
        B = params.shape[0]; ml = self.mano_layer; dtp = params.dtype
        T = state.transforms; Rg = T[:, :, :3, :3]; pg = T[:, :, :3, 3]
        theta = self._full_theta(params); Jr = so3_right_jacobian(theta); axes = Rg @ Jr        # (B,16,3,3) col m = world axis of DOF (j,m)
        betas = self._betas_for(B, params)
        shapedirs = ml.th_shapedirs.to(params); v_shaped = ml.th_v_template.to(params) + torch.einsum("vdk,bk->bvd", shapedirs, betas)
        J_rest = torch.matmul(self.J_regressor.to(params), v_shaped)                                     # (B,16,3)
        tips = self._tips; w = ml.th_weights.to(params)[tips]                                            # (5,16)
        posed = ml.th_posedirs.to(params)[tips]                                                          # (5,3,135)
        R_loc = so3_exp(theta.reshape(-1, 3)).reshape(B, 16, 3, 3)
        I3 = torch.eye(3, dtype=dtp, device=params.device)
        feat = (R_loc[:, 1:] - I3).reshape(B, 135)
        v_posed = v_shaped[:, tips] + torch.einsum("vdk,bk->bvd", posed, feat)                           # (B,5,3)
        q = pg[:, None] + torch.einsum("bkij,bvkj->bvki", Rg, v_posed[:, :, None, :] - J_rest[:, None, :, :])   # (B,5,16,3) bone contributions
        anc = self._anc.to(params)                                                                        # (16,16): anc[k,j] = 1 if j is j<=k on the chain
        m = torch.einsum("vk,kj->vj", w, anc)                                                             # (5,16)
        sq = torch.einsum("vk,kj,bvki->bvji", w, anc, q)                                                  # (5,16) weighted points per joint
        d = sq - m[None, :, :, None].permute(0, 1, 2, 3) * pg[:, None]                                    # (B,5,16,3)
        out = torch.cross(axes[:, None, :, :, :].expand(B, 5, 16, 3, 3), d[..., None].expand(B, 5, 16, 3, 3), dim=3)   # (B,5,16,3,3dof)
        dP = out.permute(0, 1, 3, 2, 4).reshape(B, 5, 3, 48)
        # corrective term: dPc/dtheta_{j,m} = posed @ d feat/d theta_{j,m};  d vec(R_j)/d delta_m = R_j [ (Jr_j)_{:,m} ]_x
        Rw = torch.einsum("vk,bkij->bvij", w, Rg)                                                         # (B,5,3,3)
        dfeat = params.new_zeros(B, 135, 48)
        for j in range(1, 16):
            for mdof in range(3):
                gen = _hat(Jr[:, j, :, mdof])                                                              # (B,3,3)
                dR = torch.matmul(R_loc[:, j], gen)                                                        # (B,3,3)
                dfeat[:, (j - 1) * 9: j * 9, j * 3 + mdof] = dR.reshape(B, 9)
        dPc = torch.einsum("vdk,bkp->bvdp", posed, dfeat)                                                  # (B,5,3,48)
        dP = dP + torch.einsum("bvij,bvjp->bvip", Rw, dPc)
        return dP

    def landmark_jacobian(self, params: Tensor, state: State) -> Optional[Tensor]:
        if state.transforms is None:
            return None  # autograd fallback
        B = params.shape[0]
        pose = params[:, :48]
        betas = self._betas_for(B, params)
        J_reg = self.J_regressor.to(params)
        T = state.transforms                                     # (B,16,4,4) uncentered
        Rg = T[:, :, :3, :3]                                     # (B,16,3,3)
        pg = T[:, :, :3, 3]                                      # (B,16,3)
        theta = self._full_theta(params)
        Jr = so3_right_jacobian(theta)                           # (B,16,3,3)
        axes = Rg @ Jr                                           # (B,16,3,3): col m = world axis for dtheta_{j,m}
        # true fingertip positions (skinned ~rigidly to the distal joint)
        shapedirs = self.mano_layer.th_shapedirs.to(params)      # (778,3,10)
        bS = torch.matmul(shapedirs, betas.transpose(0, 1)).permute(2, 0, 1)  # (B,778,3)
        v_rest = self.mano_layer.th_v_template.to(params) + bS   # (B,778,3)
        J_rest = torch.matmul(J_reg, v_rest)                     # (B,16,3)
        tips_rest = v_rest[:, self._tips]                        # (B,5,3)
        d = _TIP_DISTAL
        tip_pos = pg[:, d] + torch.einsum("bdij,bdj->bdi", Rg[:, d], tips_rest - J_rest[:, d])  # (B,5,3)
        P = torch.cat([pg, tip_pos], dim=1)                      # (B,21,3) stack order
        A = torch.cat([self._anc.to(params), self._anc.to(params)[d]], dim=0)  # (21,16)
        S = P.shape[1]
        diff = P.unsqueeze(2) - pg.unsqueeze(1)                  # (B,S,16,3)
        a_e = axes.unsqueeze(1).expand(B, S, 16, 3, 3)
        d_e = diff.unsqueeze(-1).expand(B, S, 16, 3, 3)
        cr = torch.cross(a_e, d_e, dim=3) * A.view(1, S, 16, 1, 1)
        dP = cr.permute(0, 1, 3, 2, 4).reshape(B, S, 3, 48)      # (B,21,3,48) stack order
        if self.tip_corrective and self.joints == "kinematic":
            dP = torch.cat([dP[:, :16], self._tip_jacobian(params, state)], dim=1)    # joint centres are exact rigid rows; only the tips carry skinning
        elif self.tip_corrective:
            if self._exact_idx is None:                                              # support of the joint regressor + the 5 tips
                nz = (self.J_regressor.abs() > 1e-8).any(0).nonzero().flatten().tolist()
                self._exact_idx = sorted(set(nz) | set(_TIP_VERTS))
                self._exact_pos = {v: i for i, v in enumerate(self._exact_idx)}
                self._exact_reg = self.J_regressor[:, self._exact_idx].clone()        # (16,V)
                self._exact_tip = [self._exact_pos[v] for v in _TIP_VERTS]
            dV = self._vertex_jacobian(params, state, self._exact_idx, centered=False)  # (B,V,3,48)
            dJ16 = torch.einsum("jv,bvdp->bjdp", self._exact_reg.to(params), dV)      # joints are regressed from the POSED vertices
            dP = torch.cat([dJ16, dV[:, self._exact_tip]], dim=1)
        dJc = dP[:, _OPENPOSE_PERM]                              # to openpose order
        # centre like the layer: it subtracts ITS joint `center_idx` (a skeleton centre / manotorch tip vertex), not our
        # landmark `center_idx` (with regressed joints those differ; subtracting the regressed joint's Jacobian was a
        # ~1e-3 relative error of the exact path)
        dJc = dJc - self._center_jacobian(params, state)[:, None]
        dJc = dJc[:, : self.num_joints]                          # (B,J,3,48)
        dtrans = torch.eye(3, dtype=params.dtype, device=params.device)
        dtrans = dtrans.view(1, 1, 3, 3).expand(B, self.num_joints, 3, 3)
        return torch.cat([dJc, dtrans], dim=-1)                 # (B,J,3,51)

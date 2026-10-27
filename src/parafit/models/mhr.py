"""MhrModel -- Meta's Momentum Human Rig (MHR) under the parafit Model interface. [extra: mhr]

Pure-torch re-implementation of the MHR forward (linear parameter transform ->
Euler-XYZ forward kinematics with per-joint uniform scale -> neural pose
correctives -> linear blend skinning of the identity-blendshaped rest mesh),
validated against ``mhr.mhr.MHR`` (and therefore the TorchScript release model)
to ~1e-4 cm. It is re-implemented rather than wrapped because (1) the pymomentum
FK/skinning backend is a TorchScript ``autograd.Function`` that functorch cannot
transform, so an exact ``vmap(jacrev)`` Jacobian is impossible through it, and
(2) the analytic rigid-ride Jacobian needs the per-joint global TRS and the
parameter transform as plain tensors.

Parameter vector: ``theta = [model_params(204) | identity(45)]`` = 249 DOF.
Face-expression coefficients are fixed to zero (not fitted). Units are
centimetres (MHR native). Landmarks are all 127 skeleton joint positions plus a
fixed subset of mesh vertices (farthest-point sampled at init: body + both
hands) so hands get a fair share of the correspondences.

Jacobian modes (``jac_mode``):

* ``"rigid"`` -- analytic *rigid-ride* (the MANO recipe in ``models/mano.py``):
  every landmark rides rigidly on one joint (the joint itself, or the vertex's
  dominant skinning bone). Derivatives go through the kinematic chain
  (Euler-XYZ axes, log2 scales, local translations) and the linear parameter
  transform, but NOT through the pose-corrective MLP nor the skinning-weight
  blend. The identity block is the exact skinned blendshape (linear, cheap).
* ``"lbs"`` -- like ``"rigid"`` but a vertex blends the chain-Jacobians of all
  its skinning bones (the exact LBS Jacobian, still no corrective term).
* ``"exact"`` -- autograd through everything incl. the corrective MLP, via
  chunked ``vmap(jacrev)`` (the same computation as parafit's fallback, chunked
  over the batch to bound memory).
* ``"exact_mlp"`` -- the *structured* exact Jacobian: analytic LBS chain term
  (as ``"lbs"``) plus the skinned autograd Jacobian of the corrective MLP alone
  (d delta / d theta, 750->3000->3*Vs, cheap). Numerically identical to
  ``"exact"`` (rel. err ~1e-6) at a fraction of the cost.

Import-order trap: ``pymomentum`` must be imported *before* ``torch`` in the
process, otherwise ``Character.with_blend_shape`` segfaults (pymomentum-gpu
0.1.97.post20 / torch 2.8). This module imports pymomentum lazily, so import
``pymomentum.geometry`` at the top of your script.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from parafit.core.types import ParamSpec, State
from parafit.models.base import Model
from parafit.registry import register_model

N_MODEL = 204
N_IDENTITY = 45
N_FACE = 72
_LN2 = math.log(2.0)
_HAND_TOKENS = ("thumb", "index", "middle", "ring", "pinky")


@register_model("mhr", extra="mhr")
def _build_mhr(*args, **kwargs) -> "MhrModel":
    return MhrModel(*args, **kwargs)


# ----------------------------------------------------------------------------
# small rotation helpers (torch-only, functorch-friendly)
# ----------------------------------------------------------------------------

def euler_xyz_to_mat(e: Tensor) -> Tensor:
    """(..., 3) Euler XYZ (roll, pitch, yaw) -> (..., 3, 3), R = Rz(yaw) Ry(pitch) Rx(roll).

    This is pymomentum's ``euler_xyz_to_quaternion`` convention ("first rotate
    about X, then Y, then Z") and the 9-D form of MHR's ``batch6DFromXYZ``.
    """
    a, b, c = e.unbind(-1)
    ca, cb, cc = torch.cos(a), torch.cos(b), torch.cos(c)
    sa, sb, sc = torch.sin(a), torch.sin(b), torch.sin(c)
    r00 = cb * cc
    r01 = -ca * sc + sa * sb * cc
    r02 = sa * sc + ca * sb * cc
    r10 = cb * sc
    r11 = ca * cc + sa * sb * sc
    r12 = -sa * cc + ca * sb * sc
    r20 = -sb
    r21 = sa * cb
    r22 = ca * cb
    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(*e.shape[:-1], 3, 3)


def quat_xyzw_to_mat(q: Tensor) -> Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    return torch.stack(
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
         2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
         2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1
    ).reshape(*q.shape[:-1], 3, 3)


def _farthest_point_sample(points: np.ndarray, n: int, start: int = 0) -> np.ndarray:
    """Deterministic FPS over ``points`` (N,3) -> indices (n,)."""
    N = points.shape[0]
    n = min(n, N)
    sel = np.empty(n, dtype=np.int64)
    d = np.full(N, np.inf)
    cur = start
    for i in range(n):
        sel[i] = cur
        d = np.minimum(d, ((points - points[cur]) ** 2).sum(1))
        cur = int(np.argmax(d))
    return sel


class MhrModel(Model):
    """MHR body under the parafit ``Model`` interface.

    ``theta`` layout: ``model`` (0..204) then ``identity`` (204..249).
    ``forward`` returns ``State(landmarks=(B,L,3), verts=None|(B,V,3),
    transforms=(B,J,3,5))`` where ``transforms`` packs ``[R_g | t_g | s_g]``.
    """

    def __init__(
        self,
        asset_folder: Optional[str] = None,
        lod: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        jac_mode: str = "rigid",           # "rigid" | "lbs" | "exact"
        correctives: bool = True,          # neural pose correctives in the forward
        n_vertex_landmarks: int = 200,     # total mesh-vertex landmarks
        n_hand_vertex_landmarks: int = 40, # per hand, taken from the total
        full_mesh: bool = False,           # also return the full mesh in State.verts
        exact_chunk: int = 32,             # batch chunk for the autograd Jacobian
        joint_landmarks: bool = True,
    ):
        import pymomentum.geometry as pym_geometry  # noqa: WPS433 (import before torch in the process!)
        from mhr.io import (
            get_corrective_activation_path, get_default_asset_folder, get_mhr_blendshapes_path,
            get_mhr_fbx_path, get_mhr_model_path,
        )

        folder = Path(asset_folder) if asset_folder else get_default_asset_folder()
        self.lod = lod
        self.device = torch.device(device)
        self.dtype = dtype
        self.jac_mode = jac_mode
        self.correctives = correctives
        self.full_mesh = full_mesh
        self.exact_chunk = exact_chunk
        self.joint_landmarks = joint_landmarks

        char = pym_geometry.Character.load_fbx(
            get_mhr_fbx_path(folder, lod), get_mhr_model_path(folder), load_blendshapes=True
        )
        self.character = char                       # raw 204-param character (for pymomentum baselines)
        dev, dt = self.device, dtype

        # --- skeleton ---------------------------------------------------------
        skel = char.skeleton
        self.joint_names = list(skel.joint_names)
        J = len(self.joint_names)
        self.n_joints = J
        parents = torch.as_tensor(np.asarray(skel.joint_parents), dtype=torch.long)
        self.parents = parents
        self.offsets = torch.as_tensor(np.asarray(skel.offsets), dtype=dt).to(dev)          # (J,3)
        self.R_pre = quat_xyzw_to_mat(torch.as_tensor(np.asarray(skel.pre_rotations), dtype=dt)).to(dev)  # (J,3,3)
        # depth levels for a parallel FK
        depth = [0] * J
        for j in range(J):
            p = int(parents[j])
            depth[j] = 0 if p < 0 else depth[p] + 1
        self.levels = [torch.as_tensor([j for j in range(J) if depth[j] == d], dtype=torch.long, device=dev)
                       for d in range(max(depth) + 1)]
        order = torch.cat(self.levels)
        self.pos_of = torch.empty(J, dtype=torch.long, device=dev)
        self.pos_of[order] = torch.arange(J, device=dev)
        self.order = order
        # ancestor mask A[k, j] = 1 iff joint j lies on the chain root..k (inclusive)
        anc = torch.zeros(J, J)
        for k in range(J):
            j = k
            while j >= 0:
                anc[k, j] = 1.0
                j = int(parents[j])
        self.anc = anc.to(dev)

        # --- parameter transform ------------------------------------------------
        pt = char.parameter_transform
        T = pt.transform
        T = T.to_dense() if T.is_sparse else T
        T = T.to(dtype=dt)
        assert T.shape == (J * 7, N_MODEL), T.shape
        self.T = T.to(dev)                                                             # (889,204)
        self.param_names = list(pt.names)
        self.pose_mask = pt.pose_parameters.clone().bool()
        self.rigid_mask = pt.rigid_parameters.clone().bool()
        self.scale_mask = pt.scaling_parameters.clone().bool()
        self.root_trans_mask = torch.tensor([n in ("root_tx", "root_ty", "root_tz") for n in self.param_names])
        self.root_rot_mask = torch.tensor([n in ("root_rx", "root_ry", "root_rz") for n in self.param_names])
        self.rot_mask = self.pose_mask | self.root_rot_mask                                # "rotational" params
        rows_used = (T != 0).any(1).nonzero().flatten()
        self.rows_used = rows_used.to(dev)
        self.T_used = self.T[self.rows_used]                                               # (Ru,204)
        self.row_joint = (rows_used // 7).to(dev)
        self.row_kind = (rows_used % 7).to(dev)
        # min/max limits on model params (for plausible sampling)
        lo = torch.full((N_MODEL,), -float("inf")); hi = torch.full((N_MODEL,), float("inf"))
        for lim in char.parameter_limits:
            s = repr(lim)
            if "MinMax" in s and "param=" in s:
                try:
                    idx = int(s.split("param=")[1].split(",")[0])
                    mn = float(s.split("min=")[1].split(",")[0]); mx = float(s.split("max=")[1].split(")")[0])
                    lo[idx], hi[idx] = mn, mx
                except ValueError:
                    pass
        self.param_lo, self.param_hi = lo, hi

        # --- mesh / blendshapes / skinning ------------------------------------------
        self.faces = torch.as_tensor(np.asarray(char.mesh.faces), dtype=torch.long)
        base = torch.as_tensor(np.asarray(char.blend_shape.base_shape), dtype=dt)             # (V,3)
        S = torch.as_tensor(np.asarray(char.blend_shape.shape_vectors), dtype=dt)[:N_IDENTITY]  # (45,V,3)
        V = base.shape[0]
        self.n_verts = V
        skin_idx = torch.as_tensor(np.asarray(char.skin_weights.index), dtype=torch.long)     # (V,8)
        skin_w = torch.as_tensor(np.asarray(char.skin_weights.weight), dtype=dt)               # (V,8)
        inv_bind = torch.as_tensor(np.asarray(char.inverse_bind_pose), dtype=torch.float64)   # (J,4,4)
        lin = inv_bind[:, :3, :3]
        U, Sv, Vt = torch.linalg.svd(lin)
        self.s0 = Sv[:, 0].to(dt).to(dev)                                                      # (J,)
        self.R0 = (U @ Vt).to(dt).to(dev)                                                      # (J,3,3)
        self.t0 = inv_bind[:, :3, 3].to(dt).to(dev)                                            # (J,3)
        self.base = base.to(dev)
        self.S = S.to(dev)
        self.skin_idx = skin_idx.to(dev)
        self.skin_w = skin_w.to(dev)
        self.dominant = skin_idx.gather(1, skin_w.argmax(1, keepdim=True)).squeeze(1)          # (V,) joint id of the dominant bone (cpu)

        # --- pose correctives (dense form of MHR's SparseLinear -> ReLU -> Linear) ----
        bs = np.load(get_mhr_blendshapes_path(folder, lod))
        act = np.load(get_corrective_activation_path(folder))
        comps = bs["corrective_blendshapes"]                                                   # (C,V,3)
        n_comp = comps.shape[0]
        mask = torch.as_tensor(act["posedirs_sparse_mask"])
        W1 = torch.zeros(mask.shape, dtype=dt)
        si = torch.as_tensor(act["0.sparse_indices"], dtype=torch.long)
        W1[si[0], si[1]] = torch.as_tensor(act["0.sparse_weight"], dtype=dt)
        self.W1 = W1.to(dev)                                                                   # (C=3000, 750)
        self.C = torch.as_tensor(comps.reshape(n_comp, -1), dtype=dt).to(dev)                  # (C, 3V)
        self.n_comp = n_comp

        # --- landmark vertex subset (FPS: body + hands) -----------------------------------
        hand_j = [i for i, n in enumerate(self.joint_names) if any(t in n for t in _HAND_TOKENS) or n.endswith("_wrist")]
        self.hand_joints = torch.as_tensor(hand_j, dtype=torch.long)
        is_hand_j = torch.zeros(J, dtype=torch.bool); is_hand_j[self.hand_joints] = True
        self.hand_vertex_mask = is_hand_j[self.dominant]                                       # (V,) cpu
        left_j = torch.tensor([n.startswith("l_") for n in self.joint_names])
        base_np = base.numpy()
        vh_r = ((self.hand_vertex_mask) & (~left_j[self.dominant])).nonzero().flatten().numpy()
        vh_l = ((self.hand_vertex_mask) & (left_j[self.dominant])).nonzero().flatten().numpy()
        vb = (~self.hand_vertex_mask).nonzero().flatten().numpy()
        n_body = max(n_vertex_landmarks - 2 * n_hand_vertex_landmarks, 0)
        sel = np.concatenate([
            vb[_farthest_point_sample(base_np[vb], n_body)],
            vh_r[_farthest_point_sample(base_np[vh_r], n_hand_vertex_landmarks)],
            vh_l[_farthest_point_sample(base_np[vh_l], n_hand_vertex_landmarks)],
        ])
        self.vsub = torch.as_tensor(sel, dtype=torch.long).to(dev)                              # (Vs,)
        self.n_vsub = int(self.vsub.numel())
        vs = self.vsub
        self.C_sub = self.C.reshape(n_comp, V, 3)[:, vs].reshape(n_comp, -1)                    # (C, 3Vs)
        self.base_sub = self.base[vs]
        self.S_sub = self.S[:, vs]                                                               # (45,Vs,3)
        self.skin_idx_sub = self.skin_idx[vs]
        self.skin_w_sub = self.skin_w[vs]
        self.dominant_sub = self.dominant.to(dev)[vs]
        self.landmark_is_hand = torch.cat([is_hand_j, self.hand_vertex_mask[sel]]) if joint_landmarks else self.hand_vertex_mask[sel]
        self.n_landmarks = (J if joint_landmarks else 0) + self.n_vsub
        # flattened (vertex, bone) pairs for the full mesh
        m = skin_w > 1e-5
        self.pair_v = m.nonzero()[:, 0].to(dev)
        self.pair_j = skin_idx[m].to(dev)
        self.pair_w = skin_w[m].to(dev)

        self.param_spec = ParamSpec({"model": (0, N_MODEL), "identity": (N_MODEL, N_IDENTITY)})

    # ------------------------------------------------------------------ cache (no pymomentum / MHR assets needed to load)
    _CACHE_SKIP = ("character",)

    def save_cache(self, path: str, keep_full_correctives: bool = True) -> None:
        """Serialise every attribute the forward / Jacobians use (tensors moved to CPU), so the model can be rebuilt with
        ``MhrModel.from_cache`` on a machine without pymomentum. ``C`` (3000 x 3V corrective components, ~660 MB fp32)
        is only needed for the full mesh with correctives; drop it with keep_full_correctives=False."""
        d = {}
        for k, v in self.__dict__.items():
            if k in self._CACHE_SKIP or (k == "C" and not keep_full_correctives): continue
            if torch.is_tensor(v): d[k] = v.detach().cpu()
            elif isinstance(v, list) and v and torch.is_tensor(v[0]): d[k] = [x.detach().cpu() for x in v]
            elif isinstance(v, (int, float, bool, str, list, tuple, dict, type(None))): d[k] = v
            elif isinstance(v, (torch.device, torch.dtype)): d[k] = str(v)
            elif hasattr(v, "spec"): d[k] = ("ParamSpec", v.spec)
        torch.save(d, path)

    @classmethod
    def from_cache(cls, path: str, device: str = "cpu", dtype: torch.dtype = torch.float32, jac_mode: Optional[str] = None,
                   correctives: Optional[bool] = None) -> "MhrModel":
        d = torch.load(path, map_location="cpu", weights_only=False); m = cls.__new__(cls); dev = torch.device(device)
        for k, v in d.items():
            if torch.is_tensor(v): v = v.to(dev, dtype) if v.is_floating_point() else v.to(dev)
            elif isinstance(v, list) and v and torch.is_tensor(v[0]): v = [x.to(dev) for x in v]
            elif isinstance(v, tuple) and len(v) == 2 and v[0] == "ParamSpec": v = ParamSpec(v[1])
            elif k == "device": v = dev
            elif k == "dtype": v = dtype
            setattr(m, k, v)
        m.character = None
        # cpu-side tensors the forward indexes on the host stay on cpu (as in __init__)
        for k in ("parents", "hand_joints", "hand_vertex_mask", "dominant", "faces", "pose_mask", "rigid_mask", "scale_mask", "root_trans_mask", "root_rot_mask", "rot_mask", "param_lo", "param_hi", "landmark_is_hand"):
            if hasattr(m, k) and torch.is_tensor(getattr(m, k)): setattr(m, k, getattr(m, k).cpu())
        if jac_mode is not None: m.jac_mode = jac_mode
        if correctives is not None: m.correctives = correctives
        return m

    # ------------------------------------------------------------------ helpers
    def to(self, device):
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v) and v.device == self.device:
                setattr(self, k, v.to(device))
        self.levels = [l.to(device) for l in self.levels]
        self.device = torch.device(device)
        return self

    def _joint_params(self, model: Tensor) -> Tensor:
        return (model @ self.T.T).reshape(model.shape[0], self.n_joints, 7)

    def _fk(self, jp: Tensor):
        """joint params (B,J,7) -> global R (B,J,3,3), t (B,J,3), s (B,J)."""
        Rl = self.R_pre.unsqueeze(0) @ euler_xyz_to_mat(jp[..., 3:6])
        tl = self.offsets.unsqueeze(0) + jp[..., 0:3]
        sl = torch.exp2(jp[..., 6])
        Rs, ts, ss = [], [], []
        for li, idx in enumerate(self.levels):
            if li == 0:
                Rs.append(Rl[:, idx]); ts.append(tl[:, idx]); ss.append(sl[:, idx])
                continue
            Rg_prev, tg_prev, sg_prev = torch.cat(Rs, 1), torch.cat(ts, 1), torch.cat(ss, 1)
            pp = self.pos_of[self.parents.to(idx.device)[idx]]
            Rp, tp, sp = Rg_prev[:, pp], tg_prev[:, pp], sg_prev[:, pp]
            Rs.append(Rp @ Rl[:, idx])
            ts.append(tp + sp.unsqueeze(-1) * (Rp @ tl[:, idx].unsqueeze(-1)).squeeze(-1))
            ss.append(sp * sl[:, idx])
        Rg, tg, sg = torch.cat(Rs, 1), torch.cat(ts, 1), torch.cat(ss, 1)
        inv = self.pos_of
        return Rg[:, inv], tg[:, inv], sg[:, inv]

    def _pose_feats(self, jp: Tensor) -> Tensor:
        R = euler_xyz_to_mat(jp[:, 2:, 3:6])                       # (B,125,3,3)
        f6 = torch.cat([R[..., :, 0], R[..., :, 1]], -1)            # columns 0 and 1
        f6 = f6 - torch.tensor([1.0, 0, 0, 0, 1.0, 0], dtype=f6.dtype, device=f6.device)
        return f6.flatten(1)                                        # (B,750)

    def corrective_offsets(self, jp: Tensor, subset: bool = True) -> Tensor:
        """Neural pose correctives at joint params ``jp``: (B,Vs,3) or (B,V,3)."""
        h = torch.relu(self._pose_feats(jp) @ self.W1.T)            # (B,C)
        C = self.C_sub if subset else self.C
        return (h @ C).reshape(jp.shape[0], -1, 3)

    def _joint_state(self, Rg, tg, sg):
        """G_k o Bind_k^{-1} as (R', t', s')."""
        Rp = Rg @ self.R0.unsqueeze(0)
        tp = tg + sg.unsqueeze(-1) * (Rg @ self.t0.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        sp = sg * self.s0.unsqueeze(0)
        return Rp, tp, sp

    def _skin_subset(self, Rp, tp, sp, rest):
        """LBS of the landmark-vertex subset. rest (B,Vs,3) -> (B,Vs,3), plus per-bone points (B,Vs,8,3)."""
        k = self.skin_idx_sub                                        # (Vs,8)
        P = tp[:, k] + sp[:, k].unsqueeze(-1) * torch.einsum("bvkij,bvj->bvki", Rp[:, k], rest)
        return (self.skin_w_sub.unsqueeze(-1) * P).sum(2), P

    def _skin_full(self, Rp, tp, sp, rest):
        """LBS of the full mesh via flattened (vertex,bone) pairs. rest (B,V,3) -> (B,V,3)."""
        B = rest.shape[0]
        pv, pj, pw = self.pair_v, self.pair_j, self.pair_w
        x = rest[:, pv]                                              # (B,P,3)
        y = tp[:, pj] + sp[:, pj].unsqueeze(-1) * torch.einsum("bpij,bpj->bpi", Rp[:, pj], x)
        out = torch.zeros(B, self.n_verts, 3, dtype=rest.dtype, device=rest.device)
        return out.index_add(1, pv, y * pw.view(1, -1, 1))

    def joint_transforms(self, params: Tensor):
        """Global joint transforms at ``params`` (B,249) or (B,204): rotation (B,J,3,3), translation (B,J,3) cm,
        uniform scale (B,J). A point p rigidly attached to joint k sits at ``t_k + s_k R_k p``."""
        return self._fk(self._joint_params(params[:, :N_MODEL]))

    # ------------------------------------------------------------------ Model API
    def forward(self, params: Tensor, full_mesh: Optional[bool] = None) -> State:
        B = params.shape[0]
        model, ident = params[:, :N_MODEL], params[:, N_MODEL:N_MODEL + N_IDENTITY]
        jp = self._joint_params(model)
        Rg, tg, sg = self._fk(jp)
        Rp, tp, sp = self._joint_state(Rg, tg, sg)
        rest = self.base_sub.unsqueeze(0) + torch.einsum("bn,nvd->bvd", ident, self.S_sub)
        if self.correctives:
            rest = rest + self.corrective_offsets(jp, subset=True)
        vsub, _ = self._skin_subset(Rp, tp, sp, rest)
        landmarks = torch.cat([tg, vsub], 1) if self.joint_landmarks else vsub
        transforms = torch.cat([Rg, tg.unsqueeze(-1), sg.unsqueeze(-1).unsqueeze(-1).expand(B, self.n_joints, 3, 1)], -1)
        verts = None
        if self.full_mesh if full_mesh is None else full_mesh:
            rest_f = self.base.unsqueeze(0) + torch.einsum("bn,nvd->bvd", ident, self.S)
            if self.correctives:
                rest_f = rest_f + self.corrective_offsets(jp, subset=False)
            verts = self._skin_full(Rp, tp, sp, rest_f)
        return State(landmarks=landmarks, verts=verts, transforms=transforms, batch_size=[B])

    @torch.no_grad()
    def mesh(self, params: Tensor, chunk: int = 64, correctives: Optional[bool] = None) -> Tensor:
        """Full mesh (B,V,3) in chunks (evaluation / rendering)."""
        keep = self.correctives
        if correctives is not None:
            self.correctives = correctives
        try:
            out = [self.forward(params[i:i + chunk], full_mesh=True).verts for i in range(0, params.shape[0], chunk)]
        finally:
            self.correctives = keep
        return torch.cat(out, 0)

    # ------------------------------------------------------------------ Jacobians
    def _chain_jac(self, X: Tensor, anchor: Tensor, Rg, tg, sg, jp) -> Tensor:
        """Rigid-ride chain Jacobian of sites X (B,S,3) riding on joints ``anchor`` (S,)
        w.r.t. the 204 model params: (B,S,3,204)."""
        B, S_, _ = X.shape
        rj, rk = self.row_joint, self.row_kind                       # (Ru,)
        par = self.parents.to(X.device)[rj]                          # (Ru,) parent of the row's joint
        has_par = (par >= 0)
        pc = par.clamp_min(0)
        eye = torch.eye(3, dtype=X.dtype, device=X.device)
        Rpar = torch.where(has_par.view(1, -1, 1, 1), Rg[:, pc], eye.expand(B, rj.numel(), 3, 3))
        spar = torch.where(has_par.view(1, -1), sg[:, pc], torch.ones_like(sg[:, pc]))
        # generators per used row
        e = jp[:, rj, 3:6]                                           # (B,Ru,3) Euler angles of the row's joint
        a, b, c = e.unbind(-1)
        ca, cb, cc = torch.cos(a), torch.cos(b), torch.cos(c)
        sa, sb, sc = torch.sin(a), torch.sin(b), torch.sin(c)
        # local axes: w_a = Rz Ry x, w_b = Rz y, w_c = z  (R = Rz Ry Rx)
        w_a = torch.stack([cb * cc, cb * sc, -sb], -1)
        w_b = torch.stack([-sc, cc, torch.zeros_like(sc)], -1)
        w_c = torch.stack([torch.zeros_like(sc), torch.zeros_like(sc), torch.ones_like(sc)], -1)
        Rpre_g = Rpar @ self.R_pre[rj].unsqueeze(0)                  # (B,Ru,3,3)
        kind = rk.view(1, -1, 1)
        w_loc = torch.where(kind == 3, w_a, torch.where(kind == 4, w_b, w_c))
        omega = (Rpre_g @ w_loc.unsqueeze(-1)).squeeze(-1)           # (B,Ru,3)
        # translation axes: s_par * R_par[:, m]
        m = (rk.clamp_max(2)).view(1, -1, 1, 1).expand(B, -1, 3, 1)
        t_axis = spar.unsqueeze(-1) * torch.gather(Rpar, 3, m).squeeze(-1)  # (B,Ru,3)
        is_t = (rk <= 2).view(1, 1, -1, 1).to(X.dtype)
        is_r = ((rk >= 3) & (rk <= 5)).view(1, 1, -1, 1).to(X.dtype)
        is_s = (rk == 6).view(1, 1, -1, 1).to(X.dtype)
        diff = X.unsqueeze(2) - tg[:, rj].unsqueeze(1)               # (B,S,Ru,3)
        dX = (is_t * t_axis.unsqueeze(1)
              + is_r * torch.cross(omega.unsqueeze(1).expand_as(diff), diff, dim=-1)
              + is_s * _LN2 * diff)
        A = self.anc[anchor][:, rj]                                  # (S,Ru)
        dX = dX * A.view(1, S_, -1, 1)
        return torch.einsum("bsri,rc->bsic", dX, self.T_used)        # (B,S,3,204)

    def _corrective_jac_skinned(self, model: Tensor, Rp, sp) -> Tensor:
        """Skinned Jacobian of the corrective MLP w.r.t. the 204 model params: (B,Vs,3,204)."""
        from torch.func import jacrev, vmap
        W1T, C = self.W1.T, self.C_sub

        def delta(p: Tensor) -> Tensor:                                  # (204,) -> (Vs,3) rest-frame offsets
            jp = (p @ self.T.T).reshape(1, self.n_joints, 7)
            h = torch.relu(self._pose_feats(jp) @ W1T)
            return (h @ C).reshape(-1, 3)

        dd = vmap(jacrev(delta))(model)                                  # (B,Vs,3,204)
        k = self.skin_idx_sub
        M = sp[:, k].unsqueeze(-1).unsqueeze(-1) * Rp[:, k]              # (B,Vs,8,3,3)
        M = (self.skin_w_sub.view(1, -1, 8, 1, 1) * M).sum(2)           # (B,Vs,3,3)  sum_k w_k s'_k R'_k
        return torch.einsum("bvij,bvjc->bvic", M, dd)

    def _identity_jac_sub(self, Rp, sp) -> Tensor:
        """d verts_sub / d identity: exact skinned blendshape, (B,Vs,3,45)."""
        k = self.skin_idx_sub
        M = sp[:, k].unsqueeze(-1).unsqueeze(-1) * Rp[:, k]          # (B,Vs,8,3,3)
        return torch.einsum("vk,bvkij,nvj->bvin", self.skin_w_sub, M, self.S_sub)

    def landmark_jacobian(self, params: Tensor, state: State) -> Optional[Tensor]:
        if self.jac_mode == "exact":
            return self._exact_jacobian(params)
        B = params.shape[0]
        model, ident = params[:, :N_MODEL], params[:, N_MODEL:N_MODEL + N_IDENTITY]
        jp = self._joint_params(model)
        Tm = state.transforms
        Rg, tg, sg = Tm[..., :3], Tm[..., 3], Tm[..., 0, 4]
        Rp, tp, sp = self._joint_state(Rg, tg, sg)
        J = self.n_joints
        jj = torch.arange(J, device=params.device)
        # vertices: sites
        if self.jac_mode == "rigid":
            Xv = state.landmarks[:, J:] if self.joint_landmarks else state.landmarks
            Jv = self._chain_jac(Xv, self.dominant_sub, Rg, tg, sg, jp)   # (B,Vs,3,204)
        elif self.jac_mode in ("lbs", "exact_mlp"):
            rest = self.base_sub.unsqueeze(0) + torch.einsum("bn,nvd->bvd", ident, self.S_sub)
            if self.correctives:
                rest = rest + self.corrective_offsets(jp, subset=True)
            _, P = self._skin_subset(Rp, tp, sp, rest)                    # (B,Vs,8,3)
            Vs = P.shape[1]
            Jp = self._chain_jac(P.reshape(B, Vs * 8, 3), self.skin_idx_sub.reshape(-1), Rg, tg, sg, jp)
            Jv = (self.skin_w_sub.view(1, Vs, 8, 1, 1) * Jp.reshape(B, Vs, 8, 3, N_MODEL)).sum(2)
            if self.jac_mode == "exact_mlp" and self.correctives:
                Jv = Jv + self._corrective_jac_skinned(model, Rp, sp)       # + sum_k w_k s'_k R'_k d(delta)/d(theta)
        else:
            raise ValueError(self.jac_mode)
        Jv = torch.cat([Jv, self._identity_jac_sub(Rp, sp)], -1)          # (B,Vs,3,249)
        if not self.joint_landmarks:
            return Jv
        Jj = self._chain_jac(tg, jj, Rg, tg, sg, jp)                          # (B,J,3,204)
        Jj = torch.cat([Jj, torch.zeros(B, J, 3, N_IDENTITY, dtype=Jj.dtype, device=Jj.device)], -1)
        return torch.cat([Jj, Jv], 1)

    def _exact_jacobian(self, params: Tensor) -> Tensor:
        from torch.func import jacrev, vmap

        def single(p: Tensor) -> Tensor:
            return self.forward(p.unsqueeze(0), full_mesh=False).landmarks.squeeze(0)

        f = vmap(jacrev(single))
        out = [f(params[i:i + self.exact_chunk]) for i in range(0, params.shape[0], self.exact_chunk)]
        return torch.cat(out, 0)

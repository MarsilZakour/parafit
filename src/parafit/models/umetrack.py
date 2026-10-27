"""UmeTrackModel -- UmeTrack parametric hand (FK + LBS) under the parafit Model interface. [extra: umetrack]

Ports facebookresearch/UmeTrack's ``lib/common/hand_skinning.py`` (CC-BY-NC 4.0; the hand-model json is
not vendored here, see the README for how to obtain one) to the parafit ``Model`` interface, following
the existing torch port at ``show3d_challenge/scripts/umetrack_torch.py`` / ``umetrack_hand.py``
(thesis-hope repo), validated there to 0 mm against the numpy reference and the dataset GT.

Skeleton: 22 joint slots, but only the first 20 (5 fingers x 4 DOF each, thumb/index/middle/ring/pinky
in that row order) drive forward kinematics; joints 20-21 are UmeTrack's own unused reserved slots (the
official ``_hand_skinning_transform`` only ever reads ``[:20]``) and are carried through ``theta`` as
inert, always-zero-gradient parameters. Each finger's 4 local joint transforms compose into a chain
``chain[0..3]`` (``chain[k]`` = cumulative transform after that finger's joints ``0..k``); the 17
"skinning" bone frames used for LBS are ``[root, wrist] + per-finger [chain[1], chain[2], chain[3]]``.
``chain[0]`` (the frame after only the finger's first joint) is never itself a skinning frame -- that is
an UmeTrack convention, not a bug: a point bound to a finger's ``chain[1]`` is affected by that finger's
joints ``{0,1}``, ``chain[2]`` by ``{0,1,2}``, ``chain[3]`` (the fingertip-ish frame) by all four, and a
point bound directly to the wrist frame (every "proximal"/knuckle landmark) by none of them, since a
joint's own pivot does not move under that joint's own rotation.

Handedness: UmeTrack's hand models (the generic one and per-subject profiles) are LEFT hands -- UmeTrack
indexes hand 0 = left, hand 1 = right = mirrored model. ``side="left"`` uses the file as is, ``side="right"``
mirrors it (checked anatomically: positive, within-limit flexion of the unmirrored model moves the fingertips
to the palmar side of a left hand; see ``tests/test_umetrack_model.py::test_handedness``).

Units: UmeTrack json files are millimetres; `load_umetrack_hand` converts to metres and bakes in the
file's own ``hand_scale`` (the subject's overall hand size). ``UmeTrackModel.hand_scale`` is a SEPARATE,
adjustable uniform scale (default 1.0) applied on top, around the wrist -- the loaded rest geometry is
already wrist-centred (the native ``wrist`` landmark sits at the local origin), so "around the wrist" is
a direct multiply of the rest-pose coordinates; no extra re-centring is needed.

Landmark mapping (native UmeTrack 21 -> OpenPose 21): UmeTrack's own order (``lib/common/hand.py``'s
``LANDMARK`` enum, confirmed against a SHOW3D ``profile_umetrack.json``) is::

    [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, wrist,
     thumb_intermediate, thumb_distal,
     index_proximal, index_intermediate, index_distal,
     middle_proximal, middle_intermediate, middle_distal,
     ring_proximal, ring_intermediate, ring_distal,
     pinky_proximal, pinky_intermediate, pinky_distal,
     palm_center]

Every one of these (checked via ``landmark_rest_bone_weights``) is a RIGID point on exactly one of the 17
skinning frames, never a blended one -- in particular each "proximal" landmark (the knuckle) is bound to
the WRIST frame itself. OpenPose wants 4 points per finger (CMC/MCP/IP/TIP for the thumb, MCP/PIP/DIP/TIP
for the rest); UmeTrack's thumb has only 3 native points (intermediate/distal/tip, no explicit
"thumb-base" point) while the other four fingers have exactly the 4 needed, plus an unused
``palm_center``. We therefore (a) drop ``palm_center`` (OpenPose has no palm landmark) and (b) synthesize
a "thumb CMC" point the same way UmeTrack defines the other fingers' proximal points: rigidly bound to
the wrist frame, at the rest position of the thumb's own first joint (``joint_rest_positions[0]``, its
anatomical pivot). This is the only non-literal choice in the mapping; everything else is a straight
reindex. See ``_OPENPOSE_FROM_NATIVE``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Union

import torch
from torch import Tensor

from parafit.core.lie import point_rotation_jacobian, so3_exp
from parafit.core.types import ParamSpec, State
from parafit.models.base import Model
from parafit.registry import register_model

NUM_DIGITS = 5
DOF_PER_FINGER = 4
NUM_FINGER_JOINTS = NUM_DIGITS * DOF_PER_FINGER   # 20 (of 22 joint slots)
NUM_JOINTS = 22
NUM_BONE_FRAMES = 2 + 3 * NUM_DIGITS              # 17: root, wrist, 3 per finger
NUM_MESH_VERTS = 788
NUM_NATIVE_LANDMARKS = 21

_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# Native UmeTrack 21-landmark order (facebookresearch/UmeTrack lib/common/hand.py::LANDMARK enum),
# confirmed against a SHOW3D profile_umetrack.json (see module docstring).
_NATIVE_NAMES = [
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip", "wrist",
    "thumb_intermediate", "thumb_distal",
    "index_proximal", "index_intermediate", "index_distal",
    "middle_proximal", "middle_intermediate", "middle_distal",
    "ring_proximal", "ring_intermediate", "ring_distal",
    "pinky_proximal", "pinky_intermediate", "pinky_distal",
    "palm_center",
]

# OpenPose-21 <- native UmeTrack landmark index. `None` = synthesized thumb CMC (see module docstring);
# native index 20 (palm_center) is unused.
_OPENPOSE_FROM_NATIVE = [
    5,                  # 0  wrist
    None,               # 1  thumb CMC (synthesized)
    6, 7, 0,            # 2..4  thumb MCP, IP, tip
    8, 9, 10, 1,        # 5..8  index MCP, PIP, DIP, tip
    11, 12, 13, 2,      # 9..12 middle MCP, PIP, DIP, tip
    14, 15, 16, 3,      # 13..16 ring MCP, PIP, DIP, tip
    17, 18, 19, 4,      # 17..20 pinky MCP, PIP, DIP, tip
]
assert len(_OPENPOSE_FROM_NATIVE) == NUM_NATIVE_LANDMARKS


@register_model("umetrack", extra="umetrack")
def _build_umetrack(*args, **kwargs) -> "UmeTrackModel":
    return UmeTrackModel(*args, **kwargs)


@dataclass
class UmeTrackHand:
    """Static per-subject UmeTrack hand profile (metres, radians). See `load_umetrack_hand`."""

    joint_rotation_axes: Tensor        # (22,3) unit axes, in each joint's local/chain frame
    joint_rest_positions: Tensor       # (22,3) m, rest pivot of each joint
    joint_parent: Tensor               # (22,) long, -1 = none (informational only; FK is hardcoded)
    joint_limits: Tensor               # (22,2) rad [lo, hi]
    mesh_vertices: Tensor              # (788,3) m, rest pose
    faces: Tensor                      # (F,3) long
    dense_bone_weights: Tensor         # (788,17) LBS weights of each vertex into the 17 bone frames
    landmark_rest_positions: Tensor    # (21,3) m, native order (see `_NATIVE_NAMES`)
    landmark_bone_weights: Tensor      # (21,17) dense LBS weights for the native landmarks


def load_umetrack_hand(json_path: Union[str, Path]) -> UmeTrackHand:
    """Load a UmeTrack hand-model json (mm) into an `UmeTrackHand` (m).

    Accepts the upstream ``dataset/generic_hand_model.json`` layout (flat dict) and the per-subject
    ``profile_umetrack.json`` layout used by SHOW3D (``{"hand_model": {...}}``). Not vendored here
    (CC-BY-NC 4.0, non-commercial); see the README for where to get one.
    """
    raw = json.loads(Path(json_path).read_text())
    d = raw.get("hand_model", raw)
    scale = float(d.get("hand_scale", 1.0)) / 1000.0  # bake the file's own scale + mm -> m

    def arr(key: str, dtype: torch.dtype = torch.float64) -> Tensor:
        return torch.as_tensor(d[key], dtype=dtype)

    bone_idx = arr("landmark_rest_bone_indices", torch.int64)
    bone_w = arr("landmark_rest_bone_weights")
    dense_lm_w = torch.zeros(bone_idx.shape[0], NUM_BONE_FRAMES, dtype=torch.float64)
    dense_lm_w.scatter_add_(1, bone_idx, bone_w)

    parent = arr("joint_parent", torch.int64)
    parent = torch.where(parent >= NUM_JOINTS, torch.full_like(parent, -1), parent)

    return UmeTrackHand(
        joint_rotation_axes=arr("joint_rotation_axes").float(),
        joint_rest_positions=(arr("joint_rest_positions") * scale).float(),
        joint_parent=parent,
        joint_limits=arr("joint_limits").float(),
        mesh_vertices=(arr("mesh_vertices") * scale).float(),
        faces=arr("mesh_triangles", torch.int64),
        dense_bone_weights=arr("dense_bone_weights").float(),
        landmark_rest_positions=(arr("landmark_rest_positions") * scale).float(),
        landmark_bone_weights=dense_lm_w.float(),
    )


def _mirror_hand(h: UmeTrackHand) -> UmeTrackHand:
    """Opposite-hand model: flip the y/z rotation axes and the x-coordinate of every rest position, and
    reverse the triangle winding so the mirrored mesh keeps outward normals."""
    axes = h.joint_rotation_axes.clone(); axes[:, 1:] *= -1
    rest = h.joint_rest_positions.clone(); rest[:, 0] *= -1
    lm = h.landmark_rest_positions.clone(); lm[:, 0] *= -1
    mesh = h.mesh_vertices.clone(); mesh[:, 0] *= -1
    return replace(h, joint_rotation_axes=axes, joint_rest_positions=rest,
                   landmark_rest_positions=lm, mesh_vertices=mesh, faces=h.faces.flip(-1))


def _bone_names() -> list:
    names = ["root", "wrist"]
    for f in _FINGER_NAMES:
        names += [f"{f}_j1", f"{f}_j2", f"{f}_j3"]  # after this finger's joints {0,1} / {0,1,2} / {0,1,2,3}
    return names


def _bone_parents() -> Tensor:
    """Parents of the 17 bone frames, for inspection/visualization only. The exact FK (`UmeTrackModel._fk`)
    threads an extra, unexposed frame after each finger's first joint that this cosmetic hierarchy omits
    (every finger's `_j1..3` are shown here as a simple chain off the wrist)."""
    parents = [-1, 0]
    for _ in range(NUM_DIGITS):
        base = len(parents)
        parents += [1, base, base + 1]
    return torch.tensor(parents, dtype=torch.long)


def _lbs(frames: Tensor, points: Tensor, weights: Tensor) -> Tensor:
    """Dense linear-blend skinning. ``frames`` (B,F,4,4), ``points`` (B,N,3), ``weights`` (N,F) -> (B,N,3)."""
    B, N = points.shape[:2]
    homo = torch.cat([points, points.new_ones(B, N, 1)], dim=-1)           # (B,N,4)
    posed = torch.einsum("bfij,bnj->bnfi", frames, homo)[..., :3]           # (B,N,F,3)
    return torch.einsum("nf,bnfi->bni", weights, posed)


def _eye4(B: int, dtype: torch.dtype, device) -> Tensor:
    return torch.eye(4, dtype=dtype, device=device).expand(B, 4, 4)


class UmeTrackModel(Model):
    """One UmeTrack hand under the parafit `Model` interface.

    theta (28,) = ``[root_rot(3) axis-angle | joint_angles(22) rad | root_trans(3) m]``, rotation first
    and translation last (matches `MhrHandModel` / `ManoModel`'s convention, so Kabsch-init /
    translation warm-start helpers carry over unchanged). ``ParamSpec`` blocks: ``root_rot``,
    ``joint_angles``, ``root_trans``.
    """

    def __init__(self, hand: UmeTrackHand, side: str = "right", device="cpu",
                 dtype: torch.dtype = torch.float32, with_mesh: bool = True):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self.with_mesh = with_mesh
        self.hand_scale: Union[float, Tensor] = 1.0   # separate, adjustable uniform scale (see module docstring)
        self.device, self.dtype = device, dtype

        h = hand if side == "left" else _mirror_hand(hand)     # UmeTrack hand models are LEFT hands (hand 0 = left)
        to = lambda t: t.to(device=device, dtype=dtype)
        self.axes = to(h.joint_rotation_axes)              # (22,3)
        self.rest = to(h.joint_rest_positions)              # (22,3) m
        self.joint_parent = h.joint_parent.to(device=device)
        self.joint_limits = to(h.joint_limits)              # (22,2) rad
        self.mesh_vertices = to(h.mesh_vertices)            # (788,3) m
        self.faces = h.faces.to(device=device, dtype=torch.long)
        self.dense_bone_weights = to(h.dense_bone_weights)  # (788,17)

        self.bone_names = _bone_names()
        self.bone_parents = _bone_parents().to(device=device)

        op_rest = torch.zeros(NUM_NATIVE_LANDMARKS, 3, dtype=dtype, device=device)
        op_w = torch.zeros(NUM_NATIVE_LANDMARKS, NUM_BONE_FRAMES, dtype=dtype, device=device)
        op_finger, op_depth = [], []
        for op_i, native_i in enumerate(_OPENPOSE_FROM_NATIVE):
            if native_i is None:                            # synthesized thumb CMC
                op_w[op_i, 1] = 1.0
                op_rest[op_i] = self.rest[0]                 # thumb's own first joint's rest pivot
                op_finger.append(0); op_depth.append(0)
            else:
                op_rest[op_i] = to(h.landmark_rest_positions[native_i])
                op_w[op_i] = to(h.landmark_bone_weights[native_i])
                bone = int(op_w[op_i].argmax())
                if bone <= 1:
                    op_finger.append(-1); op_depth.append(0)
                else:
                    op_finger.append((bone - 2) // 3); op_depth.append((bone - 2) % 3 + 1)
        self.op_rest = op_rest                               # (21,3) rest offsets, wrist-local
        self.op_bone_weights = op_w                          # (21,17)
        self._op_finger = op_finger                          # len-21 python list: finger idx (0..4) or -1
        self._op_depth = op_depth                            # len-21 python list: 0 (pinned) or 1/2/3

        self.num_landmarks = NUM_NATIVE_LANDMARKS
        self.param_spec = ParamSpec({"root_rot": (0, 3), "joint_angles": (3, NUM_JOINTS),
                                      "root_trans": (3 + NUM_JOINTS, 3)})

    # --- internals -------------------------------------------------------------

    def _scale(self, B: int, device, dtype: torch.dtype) -> Tensor:
        """`self.hand_scale` broadcast to (B,1,1): a python float, or a 0-dim / (B,) tensor."""
        s = torch.as_tensor(self.hand_scale, dtype=dtype, device=device).reshape(-1)
        if s.numel() == 1:
            s = s.expand(B)
        elif s.numel() != B:
            raise ValueError(f"hand_scale has {s.numel()} elements, expected 1 or B={B}")
        return s.reshape(B, 1, 1)

    def _fk(self, params: Tensor):
        """Shared forward kinematics.

        Returns ``(world_frames, local_frames, chain_local, W, rest)``: ``world_frames`` / ``local_frames``
        (B,17,4,4) are the skinning frames with / without the root pose applied; ``chain_local``
        (B,5,4,4,4) is every finger's 4 cumulative LOCAL chain transforms (``chain_local[:,f,k]`` = after
        that finger's joints ``0..k``); ``W`` (B,4,4) is the root/wrist transform; ``rest`` (B,22,3) is
        `self.rest` scaled by `self.hand_scale`.
        """
        B = params.shape[0]
        dev, dt = params.device, params.dtype
        root_rot = params[:, 0:3]
        angles = params[:, 3:3 + NUM_FINGER_JOINTS]          # (B,20); joints 20/21 are inert (module docstring)
        root_trans = params[:, 3 + NUM_JOINTS: 3 + NUM_JOINTS + 3]

        scale = self._scale(B, dev, dt)                                        # (B,1,1)
        rest = self.rest.to(dtype=dt, device=dev).unsqueeze(0) * scale         # (B,22,3)
        axes = self.axes.to(dtype=dt, device=dev)                             # (22,3)

        I4 = _eye4(B, dt, dev)
        chain_local = torch.empty(B, NUM_DIGITS, DOF_PER_FINGER, 4, 4, dtype=dt, device=dev)
        bones = [I4, I4]
        for f in range(NUM_DIGITS):
            cum = I4
            for k in range(DOF_PER_FINGER):
                j = f * DOF_PER_FINGER + k
                R = so3_exp(axes[j].expand(B, 3) * angles[:, j: j + 1])        # (B,3,3)
                t = rest[:, j] - torch.einsum("bij,bj->bi", R, rest[:, j])     # (B,3)
                L = I4.clone()
                L[:, :3, :3] = R
                L[:, :3, 3] = t
                cum = cum @ L
                chain_local[:, f, k] = cum
            bones += [chain_local[:, f, 1], chain_local[:, f, 2], chain_local[:, f, 3]]
        local_frames = torch.stack(bones, dim=1)              # (B,17,4,4)

        wrist_R = so3_exp(root_rot)
        W = I4.clone(); W[:, :3, :3] = wrist_R; W[:, :3, 3] = root_trans
        world_frames = W.unsqueeze(1) @ local_frames
        return world_frames, local_frames, chain_local, W, rest

    # --- Model API ---------------------------------------------------------------

    def forward(self, params: Tensor) -> State:
        B = params.shape[0]
        world_frames, local_frames, _, W, _ = self._fk(params)
        scale = self._scale(B, params.device, params.dtype)

        op_rest = self.op_rest.to(params).unsqueeze(0) * scale                # (B,21,3)
        local_lm = _lbs(local_frames, op_rest, self.op_bone_weights.to(params))
        wrist_R, wrist_t = W[:, :3, :3], W[:, :3, 3]
        landmarks = torch.einsum("bij,bnj->bni", wrist_R, local_lm) + wrist_t.unsqueeze(1)

        verts = None
        if self.with_mesh:
            mesh_rest = self.mesh_vertices.to(params).unsqueeze(0) * scale    # (B,788,3)
            local_v = _lbs(local_frames, mesh_rest, self.dense_bone_weights.to(params))
            verts = torch.einsum("bij,bnj->bni", wrist_R, local_v) + wrist_t.unsqueeze(1)

        return State(landmarks=landmarks, verts=verts, transforms=world_frames, batch_size=[B])

    def bone_frames(self, params: Tensor) -> Tensor:
        """(B,17,4,4) world bone transforms (metres); see `self.bone_names` / `self.bone_parents`."""
        return self._fk(params)[0]

    # --- articulated-hand interface (parafit.mapping.ManoConventionHand) -------------
    # Wrist-local, metres, zero root pose, at `self.hand_scale` (keep it 1 when wrapped: the wrapper scales).
    # The bone "frames" are UmeTrack's skinning transforms (rest space -> posed), so an attached point is
    # given by its rest-space position.

    @property
    def articulation_slice(self) -> slice:
        return self.param_spec.slice("joint_angles")

    @property
    def articulation_limits(self) -> Tensor:
        """(22,2) [lower, upper] joint-angle limits in rad (UmeTrack's own)."""
        return self.joint_limits

    def _articulated_params(self, articulation: Tensor) -> Tensor:
        params = articulation.new_zeros(articulation.shape[0], self.n_params)
        params[:, self.articulation_slice] = articulation
        return params

    def local_bone_frames(self, articulation: Tensor):
        """(B,22) joint angles -> rotation (B,17,3,3), translation (B,17,3) m, unit scale (B,17)."""
        local = self._fk(self._articulated_params(articulation))[1]
        return local[..., :3, :3], local[..., :3, 3], local.new_ones(local.shape[:2])

    def local_native_landmarks(self, articulation: Tensor) -> Tensor:
        """UmeTrack's own 21 landmarks in OpenPose order (B,21,3), wrist-local."""
        return self.forward(self._articulated_params(articulation)).landmarks

    def local_mesh(self, articulation: Tensor) -> Tensor:
        """Skinned UmeTrack mesh (B,788,3), wrist-local."""
        params = self._articulated_params(articulation)
        local = self._fk(params)[1]
        rest = self.mesh_vertices.to(params).unsqueeze(0) * self._scale(params.shape[0], params.device, params.dtype)
        return _lbs(local, rest, self.dense_bone_weights.to(params))

    def landmark_jacobian(self, params: Tensor, state: State) -> Tensor:
        """Exact d(landmarks)/d(theta), (B,21,3,28).

        Root rotation: the SO(3) right-Jacobian-corrected rigid row (`point_rotation_jacobian`), since
        every landmark is ``R_wrist @ local_point + t_wrist`` with ``local_point`` independent of the
        root. Root translation: identity. Each finger joint: a one-parameter-subgroup revolute column,
        exact with no Jacobian correction (``axis_world x (point - pivot)``, as in `TwoPartRevolute`'s
        ``art`` term), nonzero only for the (at most 3) OpenPose landmarks downstream of that joint in
        its own finger -- the "pinned" landmarks (wrist, the 4 MCPs, and the synthesized thumb CMC) have
        exactly zero joint-angle gradient by construction (rigidly bound to the wrist frame).
        """
        B = params.shape[0]
        P = self.n_params
        dev, dt = params.device, params.dtype
        _, local_frames, chain_local, W, rest = self._fk(params)
        scale = self._scale(B, dev, dt)
        op_rest = self.op_rest.to(params).unsqueeze(0) * scale
        local_lm = _lbs(local_frames, op_rest, self.op_bone_weights.to(params))   # (B,21,3)
        wrist_R = W[:, :3, :3]
        axes = self.axes.to(params)

        J = torch.zeros(B, self.num_landmarks, 3, P, dtype=dt, device=dev)
        J[..., 0:3] = point_rotation_jacobian(params[:, 0:3], local_lm)
        J[..., 3 + NUM_JOINTS: 3 + NUM_JOINTS + 3] = (
            torch.eye(3, dtype=dt, device=dev).view(1, 1, 3, 3).expand(B, self.num_landmarks, 3, 3)
        )

        I4 = _eye4(B, dt, dev)
        for f in range(NUM_DIGITS):
            idx_by_depth = {
                d: [i for i in range(self.num_landmarks) if self._op_finger[i] == f and self._op_depth[i] == d]
                for d in (1, 2, 3)
            }
            for k in range(DOF_PER_FINGER):
                affected = [i for d in (1, 2, 3) if d >= k for i in idx_by_depth[d]]
                if not affected:
                    continue
                j = f * DOF_PER_FINGER + k
                T_parent = chain_local[:, f, k - 1] if k > 0 else I4
                Rp, tp = T_parent[:, :3, :3], T_parent[:, :3, 3]
                axis_world = torch.einsum("bij,j->bi", Rp, axes[j])                   # (B,3)
                rest_j = rest[:, j]                                                   # (B,3), already scaled
                origin = torch.einsum("bij,bj->bi", Rp, rest_j) + tp                  # (B,3)
                lever = local_lm[:, affected] - origin.unsqueeze(1)                   # (B,|aff|,3)
                col = torch.cross(axis_world.unsqueeze(1).expand_as(lever), lever, dim=-1)
                col = torch.einsum("bij,bnj->bni", wrist_R, col)                      # world frame
                J[:, affected, :, 3 + j] = col
        return J

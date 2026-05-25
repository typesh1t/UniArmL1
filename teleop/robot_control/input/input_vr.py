"""VR controller input source using XrClient."""

from __future__ import annotations

import numpy as np

from .input_source import InputSource
import time
import json
import argparse
from pathlib import Path
from xrobotoolkit_teleop.common.xr_client import XrClient



class VRInput(InputSource):
    """Input from XR VR controller (e.g. PICO, Vision Pro, Meta Quest).

    Uses VRControllerReader which handles:
    - Grip button as activation gate
    - Trigger for gripper control
    - Headset-relative pose tracking
    """

    def __init__(
        self,
        pose_source: str = "right_controller",
        grip_name: str = "right_grip",
        trigger_name: str = "right_trigger",
        scale: float = 1.2,
    ):
        self._reader = VRControllerReader(
            pose_source=pose_source,
            grip_name=grip_name,
            trigger_name=trigger_name,
            scale=scale,
        )
        self._quit = False

    def get_delta(self) -> tuple[bool, np.ndarray | None, np.ndarray | None, float]:
        return self._reader.get_delta()

    def get_button(self, name: str) -> bool:
        return bool(self._reader.xr.get_button_state_by_name(name))

    @property
    def should_quit(self) -> bool:
        """Y button to quit."""
        return self.get_button("Y")

    @property
    def xr(self):
        """Access the underlying XrClient for advanced usage."""
        return self._reader.xr


def _normalize_quat_wxyz(q, eps: float = 1e-8):
    """输入 q=[w,x,y,z]，返回归一化单位四元数；不合法返回 None"""
    q = np.asarray(q, dtype=float).reshape(4)
    if not np.isfinite(q).all():
        return None
    n = float(np.linalg.norm(q))
    if n < eps:
        return None
    return q / n


def _quat_mul(a, b):
    """四元数乘法，a,b 为 [w,x,y,z]"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_diff_angle_axis(q_ref_wxyz, q_cur_wxyz, eps: float = 1e-8):
    """
    返回 angle-axis 向量（axis*angle），尽量不 NaN
    q_ref, q_cur: [w,x,y,z]
    """
    q_ref = _normalize_quat_wxyz(q_ref_wxyz, eps=eps)
    q_cur = _normalize_quat_wxyz(q_cur_wxyz, eps=eps)
    if q_ref is None or q_cur is None:
        return np.zeros(3)

    # q_err = q_ref^{-1} * q_cur
    q_err = _quat_mul(_quat_conj(q_ref), q_cur)
    q_err = _normalize_quat_wxyz(q_err, eps=eps)
    if q_err is None:
        return np.zeros(3)

    # 统一到 w>=0，避免 2π 跳变
    if q_err[0] < 0:
        q_err = -q_err

    # w = float(np.clip(q_err[0], -1.0, 1.0))
    w=q_err[0]
    v = q_err[1:]
    angle = 2.0 * np.arccos(w)

    if not np.isfinite(angle) or angle < 1e-8:
        return np.zeros(3)

    s = np.sin(angle / 2.0)
    if abs(s) < eps:
        return np.zeros(3)

    axis = v / s
    if not np.isfinite(axis).all():
        return np.zeros(3)

    return axis * angle


def yaw_from_wxyz(q_wxyz):
    """输入 wxyz，返回 yaw（绕 z）"""
    w, x, y, z = q_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))

# =============================================================================
# 可 import：VRControllerReader
# =============================================================================
def _quat_to_rotmat_wxyz(q):
    """wxyz -> 3x3 rotation matrix"""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )

def _make_transform(xyz, q_wxyz):
    T = np.eye(4, dtype=float)
    T[:3, :3] = _quat_to_rotmat_wxyz(q_wxyz)
    T[:3, 3] = xyz
    return T


def _inv_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
def _find_headset_pose(xr: XrClient):
    """尝试读取头显 pose（不同 SDK 名称可能不同）"""
    candidates = ["headset", "hmd", "head", "center_eye", "camera"]
    for name in candidates:
        try:
            p = xr.get_pose_by_name(name)
            xyz = np.array(p[:3], dtype=float)
            q = _normalize_quat_wxyz(np.array([p[6], p[3], p[4], p[5]], dtype=float))
            if q is not None and np.isfinite(xyz).all():
                return xyz, q
        except Exception:
            pass
    return None, None


class VRControllerReader:
    """
    读 XR 手柄位姿变化量（用于替换 keyboard 控制）

    - pose_source: "right_controller"/"left_controller"
    - grip_name:   "right_grip"/"left_grip"  （侧键/握把，模拟量 key value）
    - gate:        只要 grip_value != 0.0 就 active
    - 语义：
        * active 上升沿（第一次按住）-> 设 anchor
        * active 按住期间 -> 输出相对 anchor 的累计 dxyz/drot
        * 松开 -> 停止输出(None)并清空 anchor，下次再按重新计

    返回：
        active(bool), dxyz(np.ndarray|None), drot(np.ndarray|None)
    """

    def __init__(
        self,
        pose_source: str = "right_controller",
        grip_name: str = "right_grip",
        trigger_name="right_trigger",
        scale: float = 1.0,
        rotate_headset_to_world: bool = True,
    ):
        self.xr = XrClient()
        self.pose_source = pose_source
        self.grip_name = grip_name
        self.trigger_name = "right_trigger"
        self.scale = float(scale)

        self.rotate_headset_to_world = rotate_headset_to_world

        self.anchor_xyz = None
        self.anchor_quat_wxyz = None
        self.anchor_T = None

        self.anchor_head_xyz_w = None
        self.anchor_head_quat_wxyz = None

        self.anchor_yaw = 0.0
        self._active_prev = False

        # 轴映射：先保持单位映射；如需调轴，只改这里
        self.xyz_map = np.array([
            [ 0.0,  0.0,  -1.0,  ],  # x' = z
            [ -1.0,  0.0,  0.0],  # y' = -x
            [ 0.0, 1.0,  0.0],  # z' = -y
        ], dtype=float)
        self.rot_map = np.array([[0, -1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]], dtype=float)

    def _active(self) -> bool:
        v = float(self.xr.get_key_value_by_name(self.grip_name))
        return v != 0.0

    def _read_pose(self):
        """
        返回：
          xyz_local: 手柄在“当前头显局部系”下位置
          quat_local_wxyz: 手柄在“当前头显局部系”下姿态
        """
        pose = self.xr.get_pose_by_name(self.pose_source)
        c_xyz_w = np.array(pose[:3], dtype=float)
        c_q_w = _normalize_quat_wxyz(np.array([pose[6], pose[3], pose[4], pose[5]], dtype=float))
        if c_q_w is None or not np.isfinite(c_xyz_w).all():
            return None, None

        h_xyz_w, h_q_w = _find_headset_pose(self.xr)
        if h_q_w is None:
            # 兜底：没有头显pose时退化为原始控制器世界系
            return c_xyz_w, c_q_w

        # 世界->头显局部
        R_hw = _quat_to_rotmat_wxyz(h_q_w)      # headset->world
        R_wh = R_hw.T                           # world->headset
        xyz_local = R_wh @ (c_xyz_w - h_xyz_w)

        # q_local = q_head^{-1} * q_controller
        q_local = _quat_mul(_quat_conj(h_q_w), c_q_w)
        q_local = _normalize_quat_wxyz(q_local)
        if q_local is None:
            return None, None

        return xyz_local, q_local

    def _read_trigger(self) -> float:
        try:
            v = float(self.xr.get_key_value_by_name(self.trigger_name))
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.0

    def get_delta(self):
        active = self._active()
        trigger = self._read_trigger()

        if not active:
            self.anchor_xyz = None
            self.anchor_quat_wxyz = None
            self.anchor_T = None
            self.anchor_head_xyz_w = None
            self.anchor_head_quat_wxyz = None
            self.anchor_yaw = 0.0
            self._active_prev = False
            return False, None, None, trigger

        # 读手柄世界 pose
        c_xyz_w, c_q_w = self._read_controller_world_pose()
        if c_xyz_w is None or c_q_w is None:
            return True, np.zeros(3), np.zeros(3), trigger

        # 第一次按住时，冻结头显 pose
        if not self._active_prev:
            h_xyz_w, h_q_w = _find_headset_pose(self.xr)
            if h_q_w is None or h_xyz_w is None:
                # 没有头显 pose，就退化成世界系
                self.anchor_head_xyz_w = np.zeros(3, dtype=float)
                self.anchor_head_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                self.anchor_head_xyz_w = h_xyz_w.copy()
                self.anchor_head_quat_wxyz = h_q_w.copy()

            xyz, quat_wxyz = self._controller_pose_in_anchor_head_frame(c_xyz_w, c_q_w)
            if xyz is None or quat_wxyz is None:
                return True, np.zeros(3), np.zeros(3), trigger

            self.anchor_xyz = xyz.copy()
            self.anchor_quat_wxyz = quat_wxyz.copy()
            self.anchor_T = _make_transform(xyz, quat_wxyz)
            self.anchor_yaw = yaw_from_wxyz(quat_wxyz)
            self._active_prev = True
            return True, np.zeros(3), np.zeros(3), trigger

        # 后续每一帧：仍然用“冻结的头显系”表达手柄 pose
        xyz, quat_wxyz = self._controller_pose_in_anchor_head_frame(c_xyz_w, c_q_w)
        if xyz is None or quat_wxyz is None:
            return True, np.zeros(3), np.zeros(3), trigger

        cur_T = _make_transform(xyz, quat_wxyz)

        # 相对 anchor 的严格增量
        delta_T = _inv_transform(self.anchor_T) @ cur_T
        dxyz = delta_T[:3, 3] * self.scale
        drot = quat_diff_angle_axis(self.anchor_quat_wxyz, quat_wxyz)

        # 轴映射
        dxyz = self.xyz_map @ dxyz
        drot = self.rot_map @ drot

        return True, dxyz, drot, trigger
    def _controller_pose_in_anchor_head_frame(self, c_xyz_w, c_q_w):
        """
        把手柄世界 pose 转到“按下侧键那一刻冻结的头显坐标系”下
        """
        if self.anchor_head_xyz_w is None or self.anchor_head_quat_wxyz is None:
            return None, None

        R_h0w = _quat_to_rotmat_wxyz(self.anchor_head_quat_wxyz)  # head0 -> world
        R_wh0 = R_h0w.T                                           # world -> head0

        xyz_h0 = R_wh0 @ (c_xyz_w - self.anchor_head_xyz_w)

        q_h0 = self.anchor_head_quat_wxyz
        q_local = _quat_mul(_quat_conj(q_h0), c_q_w)
        q_local = _normalize_quat_wxyz(q_local)
        if q_local is None:
            return None, None
        return xyz_h0, q_local

    def _read_controller_world_pose(self):
        pose = self.xr.get_pose_by_name(self.pose_source)
        c_xyz_w = np.array(pose[:3], dtype=float)
        c_q_w = _normalize_quat_wxyz(
            np.array([pose[6], pose[3], pose[4], pose[5]], dtype=float)
        )
        if c_q_w is None or not np.isfinite(c_xyz_w).all():
            return None, None
        return c_xyz_w, c_q_w

# =============================================================================
# CLI：采样打印/写 JSONL（保留你原文件用途）
# =============================================================================
def sample_loop(xr: XrClient, args):
    out_file = None
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(p, "a", encoding="utf-8")

    interval = 1.0 / args.rate if args.rate > 0 else 0.0

    try:
        while True:
            ts = xr.get_timestamp_ns() if hasattr(xr, "get_timestamp_ns") else int(time.time() * 1e9)
            entry = {"timestamp_ns": int(ts)}

            # pose
            try:
                entry["pose"] = xr.get_pose_by_name(args.controller).tolist()
            except Exception as e:
                entry["pose_error"] = str(e)

            # key values: trigger/grip
            key_names = ["left_trigger", "right_trigger", "left_grip", "right_grip"]
            kv = {}
            for k in key_names:
                try:
                    kv[k] = float(xr.get_key_value_by_name(k))
                except Exception as e:
                    kv[k] = f"err:{type(e).__name__}"
            entry["key_values"] = kv

            # button states (SDK 限定的那几个)
            btn_names = ["A", "B", "X", "Y", "left_menu_button", "right_menu_button", "left_axis_click", "right_axis_click"]
            bs = {}
            for b in btn_names:
                try:
                    bs[b] = bool(xr.get_button_state_by_name(b))
                except Exception as e:
                    bs[b] = f"err:{type(e).__name__}"
            entry["buttons"] = bs

            line = json.dumps(entry, ensure_ascii=False)
            if args.print:
                print(line)
            if out_file:
                out_file.write(line + "\n")
                out_file.flush()

            if interval > 0:
                time.sleep(interval)

    finally:
        if out_file:
            out_file.close()


def main():
    parser = argparse.ArgumentParser(description="Read XR controller pose and inputs.")
    parser.add_argument("--controller", type=str, default="right_controller", help="pose source name")
    parser.add_argument("--rate", type=float, default=50.0, help="sample rate (Hz)")
    parser.add_argument("--print", action="store_true", help="print JSON lines to stdout")
    parser.add_argument("--output", type=str, default="", help="append JSONL output file path")
    args = parser.parse_args()

    xr = XrClient()
    sample_loop(xr, args)


if __name__ == "__main__":
    main()


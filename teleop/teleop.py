#!/usr/bin/env python3
"""Unified teleoperation entry point.

Usage:
    python teleop/teleop.py --input vr --port /dev/ttyACM1 --record --task-dir ./data/task1
    python teleop/teleop.py --input keyboard --port /dev/ttyACM0
    python teleop/teleop.py --input leader --leader-port /dev/ttyACM2

所有参数默认值定义在 robot_control/arm/config_uniarm_l1.py 的 TeleopConfig 中，
修改该文件即可永久更改默认值，CLI 参数可临时覆盖。
"""

import argparse
import logging
from pathlib import Path

from robot_control.arm.config_uniarm_l1 import UniArmL1RobotConfig, TeleopConfig
from robot_control.arm.uniarm_l1 import UniArmL1
from robot_control.input.input_keyboard import KeyboardInput
from robot_control.input.input_leader import LeaderArmInput
from robot_control.recorder import EpisodeRecorder, NullRecorder
from robot_control.teleop_runner import TeleopRunner
from image_server.opencv.configuration_opencv import OpenCVCameraConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(p: str) -> str:
    """Resolve relative paths against the script directory."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((SCRIPT_DIR / path).resolve())


def parse_cameras(camera_strs: list[str] | None) -> dict:
    """Parse camera config from ['name:id', ...] format."""
    if not camera_strs:
        return {}
    cameras = {}
    for cam_str in camera_strs:
        if ":" not in cam_str:
            logger.warning(f"Invalid camera format '{cam_str}', use 'name:id'")
            continue
        name, device_id = cam_str.split(":", 1)
        cameras[name] = OpenCVCameraConfig(
            index_or_path=Path(f"/dev/video{device_id}"),
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
        )
    return cameras


def main():
    # Load defaults from TeleopConfig (defined in config_uniarm_l1.py)
    cfg = TeleopConfig()

    # Build argparse with dataclass field values as defaults (CLI args override)
    parser = argparse.ArgumentParser(description="Unified teleoperation for UniArmL1")
    parser.add_argument("--input", "-i", choices=["vr", "keyboard", "leader"],
                        default=cfg.input, help="Input source mode")
    parser.add_argument("--port", "-p", type=str,
                        default=cfg.port, help="Serial port for follower arm")
    parser.add_argument("--leader-port", type=str,
                        default=cfg.leader_port, help="Serial port for leader arm (leader mode only)")
    parser.add_argument("--urdf", type=str,
                        default=cfg.urdf_path, help="Path to URDF file")
    parser.add_argument("--cameras", "-c", nargs="*", type=str,
                        default=cfg.cameras, help="Cameras in 'name:id' format, e.g. head:0 wrist:2")
    parser.add_argument("--no-camera", action="store_true",
                        default=cfg.no_camera, help="Disable camera display")
    parser.add_argument("--record", "-r", action="store_true",
                        default=cfg.record, help="Enable data recording")
    parser.add_argument("--task-dir", type=str,
                        default=cfg.task_dir, help="Directory for recorded data")
    parser.add_argument("--task-goal", type=str,
                        default=cfg.task_goal, help="Task goal description")
    parser.add_argument("--record-hz", type=int,
                        default=cfg.record_hz, help="Recording frequency (Hz)")
    parser.add_argument("--meshcat", action="store_true",
                        default=cfg.meshcat, help="Enable Meshcat visualization")
    parser.add_argument("--no-real-robot", action="store_true",
                        default=cfg.no_real_robot, help="Run without real robot (simulation mode)")

    args = parser.parse_args()

    # Resolve relative paths
    urdf_path = resolve_path(args.urdf)

    # Camera config
    cameras = {} if args.no_camera else parse_cameras(args.cameras)

    # Follower arm
    follower_config = UniArmL1RobotConfig(
        port=args.port,
        cameras=cameras,
        id="follower",
        urdf_path=urdf_path,
        use_vr=(args.input == "vr"),
        no_real_robot=args.no_real_robot,
    )
    follower = UniArmL1(follower_config)

    # Input source
    leader = None
    if args.input == "vr":
        from robot_control.input.input_vr import VRInput
        input_source = VRInput()
    elif args.input == "keyboard":
        input_source = KeyboardInput()
    else:  # leader
        leader_config = UniArmL1RobotConfig(
            port=args.leader_port,
            cameras={},
            id="leader",
            urdf_path=urdf_path,
        )
        leader = UniArmL1(leader_config)
        input_source = LeaderArmInput(leader)

    # Recorder
    if args.record:
        recorder = EpisodeRecorder(
            task_dir=args.task_dir,
            task_goal=args.task_goal,
            frequency=args.record_hz,
        )
    else:
        recorder = NullRecorder()

    # Runner
    runner = TeleopRunner(
        robot=follower,
        input_source=input_source,
        recorder=recorder,
        leader=leader,
        record_hz=args.record_hz,
        show_camera=not args.no_camera,
        use_meshcat=args.meshcat,
    )

    logger.info(f"Starting teleop: input={args.input}, record={'enabled' if args.record else 'disabled'}")
    runner.setup()
    runner.run()


if __name__ == "__main__":
    main()

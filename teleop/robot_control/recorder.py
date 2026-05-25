"""Recorder abstraction for teleoperation data collection."""

import abc
import os
import shutil
import numpy as np


class Recorder(abc.ABC):
    """Abstract base class for data recorders."""

    @abc.abstractmethod
    def start_episode(self) -> bool:
        """Start a new recording episode.

        Returns:
            True if episode started successfully
        """
        ...

    @abc.abstractmethod
    def stop_episode(self) -> None:
        """Stop the current recording episode."""
        ...

    @abc.abstractmethod
    def add_item(self, q_sim: np.ndarray, colors: dict, depths: dict = None) -> None:
        """Add a data item to the current episode.

        Args:
            q_sim: joint angles in sim space [arm(5), gripper(1)]
            colors: camera frames {name: np.ndarray}
            depths: depth frames (optional)
        """
        ...

    @abc.abstractmethod
    def delete_last_episode(self) -> bool:
        """Delete the most recent episode.

        Returns:
            True if deleted successfully
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    @property
    @abc.abstractmethod
    def is_recording(self) -> bool:
        """Whether currently recording."""
        ...


class EpisodeRecorder(Recorder):
    """Recorder using EpisodeWriter (JSON + images format)."""

    def __init__(self, task_dir: str, task_goal: str = "", frequency: int = 50):
        from utils.episode_writer import EpisodeWriter

        self._writer = EpisodeWriter(
            task_dir=task_dir,
            task_goal=task_goal,
            frequency=frequency,
            rerun_log=False,
        )
        self._task_dir = task_dir
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start_episode(self) -> bool:
        ok = self._writer.create_episode()
        if ok:
            self._recording = True
        return ok

    def stop_episode(self) -> None:
        self._recording = False
        self._writer.save_episode()

    def add_item(self, q_sim: np.ndarray, colors: dict, depths: dict = None) -> None:
        states = {
            "arm": {"qpos": q_sim[:5].tolist(), "qvel": [], "torque": []},
            "gripper": {"qpos": [q_sim[5]], "qvel": [], "torque": []},
        }
        actions = {
            "arm": {"qpos": q_sim[:5].tolist(), "qvel": [], "torque": []},
            "gripper": {"qpos": [q_sim[5]], "qvel": [], "torque": []},
        }
        self._writer.add_item(colors=colors, depths=depths or {}, states=states, actions=actions)

    def delete_last_episode(self) -> bool:
        if self._recording:
            return False
        if self._writer.episode_id < 0:
            return False
        ep_dir = os.path.join(self._task_dir, f"episode_{str(self._writer.episode_id).zfill(4)}")
        if os.path.exists(ep_dir):
            shutil.rmtree(ep_dir)
            self._writer.episode_id -= 1
            return True
        return False

    def close(self) -> None:
        if self._recording:
            self.stop_episode()
        self._writer.close()

def NullRecorder() -> Recorder:
    """No-op recorder for when recording is disabled."""

    class _NullRecorder(Recorder):
        def start_episode(self) -> bool:
            return True
        def stop_episode(self) -> None:
            pass
        def add_item(self, q_sim: np.ndarray, colors: dict, depths: dict = None) -> None:
            pass
        def delete_last_episode(self) -> bool:
            return False
        def close(self) -> None:
            pass
        @property
        def is_recording(self) -> bool:
            return False

    return _NullRecorder()
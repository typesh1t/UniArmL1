"""Input source abstraction for teleoperation.

Each input source provides either:
- get_delta() -> (active, dxyz, drot, trigger) for IK-based control
- get_q_sim() -> joint angles for direct joint control (e.g. leader arm)
"""

import abc
import numpy as np


class InputSource(abc.ABC):
    """Abstract base class for teleoperation input sources.

    Subclasses must implement either get_delta() (for IK-based control
    like VR/keyboard) or get_q_sim() (for direct joint control like
    leader arm). The default get_q_sim() returns None, indicating
    this source operates in IK/delta mode.
    """

    @abc.abstractmethod
    def get_delta(self) -> tuple[bool, np.ndarray | None, np.ndarray | None, float]:
        """Read the current input delta (for IK-based control).

        Returns:
            active: whether the input source is actively controlling
            dxyz: position delta [dx, dy, dz] or None if inactive
            drot: rotation delta [rx, ry, rz] (angle-axis) or None if inactive
            trigger: gripper trigger value in [0, 1]
        """
        ...

    def get_q_sim(self) -> np.ndarray | None:
        """Read joint angles directly (for joint-based control, e.g. leader arm).

        Returns:
            Joint angles in sim space, or None if this source
            operates in IK/delta mode (default).
        """
        return None

    @abc.abstractmethod
    def get_button(self, name: str) -> bool:
        """Read a button state by name.

        Args:
            name: button name (e.g. "A", "B", "X", "Y")

        Returns:
            True if the button is currently pressed
        """
        ...

    def close(self) -> None:
        """Release resources. Override if needed."""
        pass

    @property
    def should_quit(self) -> bool:
        """Whether the user has requested to quit. Override if needed."""
        return False

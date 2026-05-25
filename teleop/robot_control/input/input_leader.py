"""Leader arm input source for direct joint control.

Controls (terminal keys):
    s   : Toggle recording
    d   : Delete last episode
    h   : Reset to home
    q   : Quit
"""

from __future__ import annotations

import sys
import select
import termios

import numpy as np

from .input_source import InputSource
from ..arm.uniarm_l1 import UniArmL1


class LeaderArmInput(InputSource):
    """Input from a leader UniArmL1 that directly provides joint angles.

    Instead of computing delta poses and running IK, this source
    reads the leader arm's current joint angles and passes them
    directly to the follower arm.

    Terminal keys are used for recording control:
        s = toggle recording, d = delete episode, h = reset home, q = quit
    """

    def __init__(self, leader: UniArmL1):
        self.leader = leader

        self._old_settings = None
        self._started = False
        self._quit = False

        self._buttons: dict[str, bool] = {
            "A": False,  # record toggle
            "B": False,  # delete episode
            "X": False,  # reset home
        }

    def _ensure_terminal(self) -> None:
        if self._started:
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        # Set raw input mode but preserve output processing (OPOST)
        # so that \n in stdout still becomes \r\n automatically
        new = termios.tcgetattr(sys.stdin)
        new[0] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON)
        new[3] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
        new[6][termios.VMIN] = 1
        new[6][termios.VTIME] = 0
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new)
        self._started = True

    def get_delta(self) -> tuple[bool, np.ndarray | None, np.ndarray | None, float]:
        # Leader arm operates in joint mode, not IK/delta mode
        return False, None, None, 0.0

    def get_q_sim(self) -> np.ndarray | None:
        self._ensure_terminal()
        # Reset one-shot buttons each frame
        self._buttons = {"A": False, "B": False, "X": False}

        key = self._read_key()
        if key is not None:
            if key == "q":
                self._quit = True
            elif key == "s":
                self._buttons["A"] = True
            elif key == "d":
                self._buttons["B"] = True
            elif key == "h":
                self._buttons["X"] = True

        return self.leader.get_cur_q_sim()

    def get_button(self, name: str) -> bool:
        return self._buttons.get(name, False)

    @property
    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        if self._started and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._started = False

    @staticmethod
    def _read_key() -> str | None:
        if select.select([sys.stdin], [], [], 0.05)[0]:
            ch = sys.stdin.read(1)
            # Consume ESC sequences
            if ch == "\x1b":
                while select.select([sys.stdin], [], [], 0.01)[0]:
                    sys.stdin.read(1)
                return None
            return ch
        return None

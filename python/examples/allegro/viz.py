"""Recording the rollout as it happens.

There is no ``render(trajectory)`` here, because there is nothing to replay: the
Allegro task is a closed-loop MPC rollout against MuJoCo, and its trajectory file
records what the simulator did rather than a plan you can play back. The picture
has to be taken while the loop runs, which is what this recorder does -- pass
``--video out.gif`` to ``main.py``, or ``--render`` for the interactive viewer.

Requires the ``sim`` extra (MuJoCo) plus Pillow.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

__all__ = ["GifRecorder"]


class GifRecorder:
    """Small MuJoCo offscreen recorder, with no dependency beyond Pillow."""

    def __init__(self, model, path, *, camera: str, width: int, height: int, fps: int):
        import mujoco

        path = Path(path)
        if path.suffix.lower() != ".gif":
            raise ValueError(f"video must be a .gif path, got {path}")
        self.path = path
        self.camera = camera
        self.fps = fps
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.frames: List[np.ndarray] = []

    def capture(self, data) -> None:
        self.renderer.update_scene(data, camera=self.camera)
        self.frames.append(np.array(self.renderer.render(), copy=True))

    def save(self) -> None:
        from PIL import Image

        if not self.frames:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(frame) for frame in self.frames]
        images[0].save(self.path, save_all=True, append_images=images[1:],
                       duration=max(1, round(1000 / self.fps)), loop=0)
        print(f"video saved to: {self.path}")

    def close(self) -> None:
        self.renderer.close()

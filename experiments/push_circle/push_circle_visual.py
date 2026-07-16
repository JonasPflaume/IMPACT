#!/usr/bin/env python3
"""
Disk-pushing trajectory visualizer + GIF generator.

Reads a trajectory written by either push-circle shooting driver and renders both
an animated GIF and a static trajectory overlay. The interesting bit to watch is the
pusher: it starts on the lower-left (where the disk's goal is), has to travel all
the way around the disk to the far side, and only then pushes the disk down into
the goal -- the maneuver that a local method initialized at the start cannot find.

Usage:
    python3 push_circle_visual.py [trajectory.txt] [--fps 25] [--stride 1]
                                  [--out <path.gif>] [--minimal]

State columns : qx qy sx sy   (disk center, pusher point)
Control columns: fn ft vx vy  (normal force, tangential friction, pusher velocity)
"""

import argparse
import os
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))
from visual_utils import resolve_output_dir, resolve_trajectory

import numpy as np
import matplotlib as mpl
if not os.environ.get("DISPLAY"):
    mpl.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


def parse_trajectory(path):
    """Return (radius, goal[4], X[T,4], U[T-1,4]) from a solver trajectory file."""
    with open(path) as f:
        lines = f.read().splitlines()

    def after(tag, n=1):
        for i, l in enumerate(lines):
            if l.startswith(tag):
                return i + n
        return None

    radius = float(lines[after("# Disk Radius")])
    goal = np.array([float(v) for v in lines[after("# Goal State")].split()])

    si = after("# State Trajectory")
    ci = next(i for i, l in enumerate(lines) if l.startswith("# Control Trajectory"))
    X = np.array([[float(v) for v in lines[i].split()]
                  for i in range(si, ci) if lines[i].strip() and not lines[i].startswith("#")])
    U = np.array([[float(v) for v in lines[i].split()]
                  for i in range(ci + 1, len(lines)) if lines[i].strip() and not lines[i].startswith("#")])
    return radius, goal, X, U


def fading_line(ax, pts, base_color, lw=2.0, zorder=2):
    """A polyline whose alpha fades from faint (old) to solid (recent)."""
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    rgba = np.zeros((n, 4))
    rgba[:, :3] = mpl.colors.to_rgb(base_color)
    rgba[:, 3] = np.linspace(0.25, 0.95, n) if n else 0.9
    lc = LineCollection(segs, colors=rgba, linewidths=lw, zorder=zorder)
    ax.add_collection(lc)
    return lc


def visualize(path, fps=25, stride=1, out=None, minimal=False):
    R, goal, X, U = parse_trajectory(path)
    q = X[:, 0:2]          # disk center
    s = X[:, 2:4]          # pusher point
    fn = U[:, 0]           # normal force
    ft = U[:, 1]           # tangential friction force
    T = X.shape[0]
    goal_q = goal[0:2]

    c_disk, c_push, c_goal, c_force = "#2f7bd6", "#e64a19", "#00a86b", "#8e24aa"

    # Axis limits from all geometry, with padding.
    allpts = np.vstack([q, s, goal_q[None, :]])
    pad = R + 0.25
    xmin, xmax = allpts[:, 0].min() - pad, allpts[:, 0].max() + pad
    ymin, ymax = allpts[:, 1].min() - pad, allpts[:, 1].max() + pad
    span = max(xmax - xmin, ymax - ymin)
    fig, ax = plt.subplots(figsize=(8, 8 * (ymax - ymin) / (xmax - xmin)), facecolor="white")
    ax.set_aspect("equal", "box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if minimal:
        ax.axis("off")
    else:
        ax.grid(True, alpha=0.3, ls="--", lw=0.5, color="#cccccc")
        ax.set_facecolor("#fafafa")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title("IMPACT — quasi-static disk push",
                     fontsize=13, fontweight="bold", color="#333")

    # Goal footprint (where the disk should end up) and its center.
    ax.add_patch(Circle(goal_q, R, fill=False, ls="--", lw=2.0, ec=c_goal, alpha=0.9, zorder=1))
    ax.scatter(*goal_q, s=60, marker="X", color=c_goal, zorder=6, label="disk goal")
    # Pusher start (== goal location); highlights why it is a hard init.
    ax.scatter(*s[0], s=90, marker="*", color=c_push, edgecolors="#5a1500",
               linewidths=1.2, zorder=7, label="pusher start")

    # Static faint trails (grow via set_segments during animation).
    disk_trail = fading_line(ax, q[:1], c_disk, lw=2.2, zorder=2)
    push_trail = fading_line(ax, s[:1], c_push, lw=1.8, zorder=2)

    # Moving artists.
    disk_patch = Circle(q[0], R, facecolor=c_disk, edgecolor="#1c4e8a",
                        lw=2.0, alpha=0.55, zorder=4, label="disk")
    ax.add_patch(disk_patch)
    push_dot = ax.scatter(*s[0], s=70, color=c_push, edgecolors="#5a1500",
                          linewidths=1.2, zorder=8, label="pusher")
    force_q = ax.quiver([s[0, 0]], [s[0, 1]], [0.0], [0.0], angles="xy",
                        scale_units="xy", scale=1.0, color=c_force, width=0.010,
                        zorder=9, alpha=0.9)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=10, family="monospace", color="#222",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
    if not minimal:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    force_scale = 0.35 * span / max(1e-6, np.abs(fn).max() + np.abs(ft).max())

    def update(frame):
        i = min(frame * stride, T - 1)
        disk_patch.center = (q[i, 0], q[i, 1])
        push_dot.set_offsets([s[i]])
        disk_trail.set_segments(np.stack([q[:i, :], q[1:i + 1, :]], axis=1) if i >= 1 else np.empty((0, 2, 2)))
        push_trail.set_segments(np.stack([s[:i, :], s[1:i + 1, :]], axis=1) if i >= 1 else np.empty((0, 2, 2)))

        # Contact force arrow (world frame) applied at the disk boundary point.
        j = min(i, U.shape[0] - 1)
        d = s[i] - q[i]
        nrm = np.hypot(*d) + 1e-9
        n_out = d / nrm
        t_hat = np.array([-n_out[1], n_out[0]])
        contact_pt = q[i] + R * n_out
        F = (-fn[j] * n_out + ft[j] * t_hat) * force_scale
        force_q.set_offsets([contact_pt])
        force_q.set_UVC(F[0], F[1])

        dist = np.hypot(*(q[i] - goal_q))
        gap = nrm - R
        hud.set_text(f"t = {i:3d}/{T-1}\n"
                     f"disk->goal = {dist:5.3f} m\n"
                     f"gap        = {gap:+5.3f} m\n"
                     f"|f_n|={fn[j]:+.2f}  f_t={ft[j]:+.2f}")
        return disk_patch, push_dot, disk_trail, push_trail, force_q, hud

    nframes = (T + stride - 1) // stride
    anim = animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=1000 / fps, blit=False, repeat=True)

    if out is None:
        out = os.path.splitext(path)[0] + ".gif"
    Path(out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving GIF ({nframes} frames @ {fps} fps) -> {out}")
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    print(f"[ok] {out}")

    # Static overlay PNG: disk snapshots + the full disk-center and pusher paths,
    # so the "travel around, then push" maneuver is visible in a single frame.
    overlay = os.path.splitext(out)[0] + "_overlay.png"
    disk_patch.set_alpha(0.0); push_dot.set_alpha(0.0); force_q.set_alpha(0.0)
    disk_patch.set_label("_nolegend_"); push_dot.set_label("_nolegend_")
    disk_trail.set_segments(np.empty((0, 2, 2))); push_trail.set_segments(np.empty((0, 2, 2)))
    hud.set_text("")
    for i in range(0, T, max(1, T // 10)):
        a = 0.12 + 0.5 * (i / (T - 1))
        ax.add_patch(Circle(q[i], R, facecolor=(0.18, 0.48, 0.84, 0.10 * a),
                            edgecolor=(0.18, 0.48, 0.84, a), lw=1.2, zorder=3))
    ax.plot(q[:, 0], q[:, 1], "-", color=c_disk, lw=2.2, alpha=0.9, zorder=5, label="disk path")
    ax.plot(s[:, 0], s[:, 1], "-", color=c_push, lw=2.0, alpha=0.9, zorder=5, label="pusher path")
    # Arrowheads along the pusher path show the go-around direction.
    for i in range(0, T - 1, max(1, T // 12)):
        ax.annotate("", xy=s[i + 1], xytext=s[i],
                    arrowprops=dict(arrowstyle="-|>", color=c_push, lw=0, alpha=0.9))
    if not minimal:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.savefig(overlay, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[ok] {overlay}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", help="trajectory file (default: newest result)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out", help="GIF path (default: results/push_circle/push_circle.gif)")
    parser.add_argument("--minimal", action="store_true")
    args = parser.parse_args()

    traj = resolve_trajectory("push_circle", args.trajectory)
    default_out = resolve_output_dir("push_circle") / "push_circle.gif"
    out = Path(args.out).expanduser().resolve() if args.out else default_out
    visualize(traj, fps=args.fps, stride=args.stride, out=out, minimal=args.minimal)

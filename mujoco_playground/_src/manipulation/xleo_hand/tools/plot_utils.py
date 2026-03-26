"""Shared plotting utilities for massage trajectory data.

Provides grouped subplot plots used by generate_massage_data.py,
replay_massage.py, and resample_massage_data.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mujoco_playground._src.manipulation.xleo_hand import constants as consts

# ---------------------------------------------------------------------------
# Semantic grouping derived from constants
# ---------------------------------------------------------------------------

# Each entry: (subplot_title, column_indices, column_labels)
JOINT_GROUPS = [
    (title, [idx for idx, _ in joints], [lbl for _, lbl in joints])
    for title, joints in consts.JOINT_GROUPS
]

TRACKED_BODY_NAMES = list(consts.TRACKED_BODY_NAMES)
KEY_BODY_NAMES = list(consts.KEY_BODY_NAMES)

# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------


def plot_grouped(
    arr: np.ndarray,
    times: np.ndarray,
    title: str,
    ylabel: str,
    groups: list[tuple[str, list[int], list[str]]],
    save_path: str,
    colors: list[str] | None = None,
) -> None:
  """Plot a (T, D) array as grouped subplots.

  Each group gets one subplot row. Multiple columns within a group are
  overlaid as separate colored lines with a legend.

  Args:
    arr: (T, D) data array.
    times: (T,) time axis in seconds.
    title: Overall figure title.
    ylabel: Y-axis label for every subplot.
    groups: list of (group_title, column_indices, column_labels).
    save_path: Where to save the figure.
    colors: Optional per-line colors (applied cyclically within each group).
  """
  n_groups = len(groups)
  fig, axes = plt.subplots(
      n_groups, 1, figsize=(14, 3.2 * n_groups), squeeze=False, sharex=True
  )
  for g, (group_title, col_ids, col_labels) in enumerate(groups):
    ax = axes[g, 0]
    for idx, col in enumerate(col_ids):
      kwargs = {"linewidth": 1.5, "label": col_labels[idx]}
      if colors is not None:
        kwargs["color"] = colors[idx % len(colors)]
      ax.plot(times, arr[:, col], **kwargs)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(group_title, fontsize=14, loc="left", fontweight="bold")
    ax.legend(fontsize=10, ncol=len(col_ids), loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
  axes[-1, 0].set_xlabel("Time (s)", fontsize=12)
  fig.suptitle(title, fontsize=20, fontweight="bold")
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  fig.savefig(save_path, dpi=150)
  plt.close(fig)
  print(f"  Saved plot: {save_path}")


def plot_3d(
    arr: np.ndarray,
    times: np.ndarray,
    title: str,
    ylabel: str,
    body_names: list[str] | None,
    dim_labels: list[str] | None,
    save_path: str,
    colors: list[str] | None = None,
) -> None:
  """Plot a (T, N_bodies, D) array as grouped subplots.

  Thin wrapper around ``plot_grouped``: reshapes (T, N, D) → (T, N*D) and
  builds one group per body with D lines each.

  Args:
    arr: (T, N, D) data array.
    times: (T,) time axis in seconds.
    title: Overall figure title.
    ylabel: Y-axis label for every subplot.
    body_names: Names for each body (subplot titles). None → ``body 0``, etc.
    dim_labels: Labels for each dimension (legend entries). None → ``d0``, etc.
    save_path: Where to save the figure.
    colors: Optional per-dimension colors (e.g. ["red", "green", "blue"]).
  """
  n_bodies = arr.shape[1]
  n_dims = arr.shape[2]
  if dim_labels is None:
    dim_labels = [f"d{d}" for d in range(n_dims)]
  if body_names is None:
    body_names = [f"body {b}" for b in range(n_bodies)]

  # Reshape (T, N, D) → (T, N*D) and build groups.
  flat = arr.reshape(arr.shape[0], -1)  # (T, N*D)
  groups = []
  for b in range(n_bodies):
    col_ids = list(range(b * n_dims, (b + 1) * n_dims))
    groups.append((body_names[b], col_ids, list(dim_labels)))

  plot_grouped(flat, times, title, ylabel, groups, save_path, colors=colors)


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def plot_all(data: dict, output_dir: Path, prefix: str = "") -> None:
  """Plot all trajectory fields in *data* and save to *output_dir*.

  Supports qpos, qvel, tracked_body_xpos, key_body_xpos, and contact_force.

  Args:
    data: Trajectory dict with at least ``qpos``, ``data_freq``.
    output_dir: Directory to write PNG files into (created if needed).
    prefix: Optional filename prefix (e.g. ``"120hz_"``).
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  freq = float(data["data_freq"])
  T = data["qpos"].shape[0]
  times = np.arange(T) / freq

  print("Generating plots...")

  # 1. qpos
  plot_grouped(
      data["qpos"],
      times,
      title=f"Joint Positions (qpos) — {T} frames @ {freq} Hz",
      ylabel="Position (rad/m)",
      groups=JOINT_GROUPS,
      save_path=str(output_dir / f"{prefix}qpos.png"),
  )

  # 2. qvel
  plot_grouped(
      data["qvel"],
      times,
      title=f"Joint Velocities (qvel) — {T} frames @ {freq} Hz",
      ylabel="Velocity (rad·s⁻¹ / m·s⁻¹)",
      groups=JOINT_GROUPS,
      save_path=str(output_dir / f"{prefix}qvel.png"),
  )

  # XYZ / force dimension colors: x=red, y=green, z=blue.
  _xyz_colors = ["red", "green", "blue"]

  # 3. tracked_body_xpos
  if "tracked_body_xpos" in data:
    tracked_names = data.get("tracked_body_names", TRACKED_BODY_NAMES)
    plot_3d(
        data["tracked_body_xpos"],
        times,
        title=f"Tracked Body Positions (xpos) — {T} frames @ {freq} Hz",
        ylabel="Position (m)",
        body_names=tracked_names,
        dim_labels=["x", "y", "z"],
        save_path=str(output_dir / f"{prefix}tracked_body_xpos.png"),
        colors=_xyz_colors,
    )

  # 4. key_body_xpos
  if "key_body_xpos" in data:
    key_names = data.get("key_body_names", KEY_BODY_NAMES)
    plot_3d(
        data["key_body_xpos"],
        times,
        title=f"Key Body Positions (xpos) — {T} frames @ {freq} Hz",
        ylabel="Position (m)",
        body_names=key_names,
        dim_labels=["x", "y", "z"],
        save_path=str(output_dir / f"{prefix}key_body_xpos.png"),
        colors=_xyz_colors,
    )

  # 5. tracked_contact_force (synthesised)
  if "tracked_contact_force" in data:
    cf_names = data.get(
        "contact_body_names", data.get("contact_sensor_names", None)
    )
    n_dims = (
        data["tracked_contact_force"].shape[2]
        if data["tracked_contact_force"].ndim == 3
        else 0
    )
    if n_dims == 3:
      dim_labels = ["fx", "fy", "fz"]
      src_label = "synthesised"
    else:
      dim_labels = ["fx", "fy", "fz", "tx", "ty", "tz"]
      src_label = "synthesised"
    plot_3d(
        data["tracked_contact_force"],
        times,
        title=f"Contact Forces ({src_label}) — {T} frames @ {freq} Hz",
        ylabel="Force (N)",
        body_names=cf_names,
        dim_labels=dim_labels,
        save_path=str(output_dir / f"{prefix}tracked_contact_force.png"),
        colors=_xyz_colors,
    )

  # 6. cfrc_ext_raw (raw cfrc_ext from simulation)
  if "cfrc_ext_raw" in data:
    cf_names = data.get(
        "contact_body_names", data.get("contact_sensor_names", None)
    )
    plot_3d(
        data["cfrc_ext_raw"],
        times,
        title=f"Raw cfrc_ext — {T} frames @ {freq} Hz",
        ylabel="Force (N)",
        body_names=cf_names,
        dim_labels=["fx", "fy", "fz"],
        save_path=str(output_dir / f"{prefix}cfrc_ext_raw.png"),
        colors=_xyz_colors,
    )

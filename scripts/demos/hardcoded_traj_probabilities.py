"""
Mobile-robot / "human" path-planning demo
========================================
• Theta* for global planning
• Bézier smoothing
• Real-time probability that the human is heading for each goal
• NEW: the marker now *follows its planned path* instead of flying straight
• UPDATED: probabilities now sum to 1.0
• FIXED: unlikely goals decay quickly thanks to lower baselines + temperature
"""
import math
from typing import List, Tuple, Dict
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scripts.dataset.synthetic.utils.utils import (
    Grid,
    spawn_goals,
    theta_star,
    smooth_path_with_beziers,
)
from scripts.dataset.synthetic.utils.plot import create_padded

# ── Workspace limits ────────────────────────────────────────────
X_MIN, Y_MIN, X_MAX, Y_MAX = -5.25, -3.5, 5.35, 9.0
ASSEMBLY_BOUNDS   = {"x_bounds": (-0.30, 0.80),  "y_bounds": (-0.20, -0.15)}
TOOL_STATION_BOUNDS = {"x_bounds": (-0.20, 0.70), "y_bounds": (0.94, 1.06)}
RANDOM_BOUNDS     = {"x_bounds": (-5.25, 5.35),  "y_bounds": (-3.50, 9.00)}
AGV_PHASE2_BOUNDS = {"x_bounds": (-2.5, -2.5),   "y_bounds": (-0.5, 2)}
AGV_POSITIONS     = [(-4.5, 4.5), (-4.5, -0.5)]
MOBILE_POSITIONS  = [(-0.625, -2.245), (-2.2, 0.5)]

# ── Animation parameters ────────────────────────────────────────
ANIMATION_SPEED = 0.1        # m/s
FPS             = 30         # frames per second
FRAMES          = 1_000

# ── Probability-model parameters ────────────────────────────────
TEMPERATURE = 3.0            # >1 sharpens, <1 flattens the distribution
WEIGHTS = {"alignment": 0.25, "distance": 0.35,
           "priority": 0.10, "consistency": 0.30}

# ----------------------------------------------------------------
#  helper: probability that human heads for a specific goal
# ----------------------------------------------------------------
def calculate_path_probability(
    human_pos: Tuple[float, float],
    human_velocity: Tuple[float, float],
    goal_pos: Tuple[float, float],
    goal_type: str,
    path_length: float,
    history: List[Tuple[float, float]],
) -> float:
    """Heuristic probability score (raw, before temperature & normalisation)."""

    # 1. velocity-to-goal alignment  ───────────────────────────────────────
    v_mag = math.hypot(*human_velocity)
    goal_vec = (goal_pos[0] - human_pos[0], goal_pos[1] - human_pos[1])
    goal_dist = math.hypot(*goal_vec)

    if v_mag > 0 and goal_dist > 0:
        v_norm = (human_velocity[0] / v_mag, human_velocity[1] / v_mag)
        goal_dir = (goal_vec[0] / goal_dist, goal_vec[1] / goal_dist)
        dot = v_norm[0] * goal_dir[0] + v_norm[1] * goal_dir[1]  # ∈ [-1, 1]
        alignment = max(0.0, dot)        # reward only when pointing towards goal
    else:
        alignment = 0.0                  # no information yet

    # 2. path length (shorter ⇒ higher score)  ─────────────────────────────
    max_d      = math.hypot(X_MAX - X_MIN, Y_MAX - Y_MIN)
    distance_s = 1 - (path_length / max_d)     # already 0-1

    # 3. goal priority  ────────────────────────────────────────────────────
    priority_s = {"tool_station": 0.9, "battery": 0.7, "agv": 0.6}.get(goal_type, 0.4)

    # 4. consistency (distance must shrink)  ───────────────────────────────
    if len(history) >= 5:
        d0 = math.hypot(history[-5][0] - goal_pos[0], history[-5][1] - goal_pos[1])
        d1 = math.hypot(human_pos[0]   - goal_pos[0], human_pos[1]   - goal_pos[1])
        consistency = max(0.0, (d0 - d1) / d0) if d0 else 1.0
    else:
        consistency = 0.0

    return (
        WEIGHTS["alignment"]   * alignment   +
        WEIGHTS["distance"]    * distance_s  +
        WEIGHTS["priority"]    * priority_s  +
        WEIGHTS["consistency"] * consistency
    )

# ----------------------------------------------------------------
#  Animation wrapper
# ----------------------------------------------------------------
class PathProbabilityAnimation:
    """Same behaviour as before, but now spawns SEVEN independent 'random' goals."""

    def __init__(self, ax, grid, pads):
        self.ax   = ax
        self.grid = grid
        self.pads = pads

        # ── Spawn human -----------------------------------------------------
        self.human_pos     = spawn_goals(1, grid, pads, (-3, -2), (7, 8))[0]
        self.human_history = [self.human_pos]
        self.human_vel     = (0, 0)

        # ── Spawn goals -----------------------------------------------------
        # Structured (single-instance) goals
        self.goals: Dict[str, Tuple[float, float]] = {
            "tool_station": spawn_goals(
                1, grid, pads,
                TOOL_STATION_BOUNDS["x_bounds"], TOOL_STATION_BOUNDS["y_bounds"]
            )[0],
            "battery": spawn_goals(
                1, grid, pads,
                ASSEMBLY_BOUNDS["x_bounds"], ASSEMBLY_BOUNDS["y_bounds"]
            )[0],
            "agv": spawn_goals(
                1, grid, pads,
                AGV_PHASE2_BOUNDS["x_bounds"], AGV_PHASE2_BOUNDS["y_bounds"]
            )[0],
        }

        # Seven purely random goals scattered in the workspace
        random_positions = spawn_goals(
            7, grid, pads,
            RANDOM_BOUNDS["x_bounds"], RANDOM_BOUNDS["y_bounds"]
        )
        for i, pos in enumerate(random_positions, 1):
            self.goals[f"random{i}"] = pos

        # ── Plot handles ----------------------------------------------------
        self.human_marker, = ax.plot(*self.human_pos, "ko", ms=10, label="Human")

        base_random_colours = [
            "tab:green", "tab:olive", "tab:cyan",
            "tab:purple", "tab:pink", "tab:brown", "tab:gray"
        ]
        colors = {
            "tool_station": "red",
            "battery": "blue",
            "agv": "orange",
            **{f"random{i+1}": base_random_colours[i] for i in range(7)},
        }
        offsets = {
            "tool_station": (0.5,  0.3),
            "battery":      (0.5, -0.1),
            "agv":          (-0.8, 0.3),
            **{f"random{i+1}": (0.5, 0.3) for i in range(7)},
        }

        self.goal_markers: Dict[str, plt.Line2D] = {}
        self.path_lines:   Dict[str, plt.Line2D] = {}
        self.prob_texts:   Dict[str, plt.Text]   = {}

        for g, pos in self.goals.items():
            self.goal_markers[g], = ax.plot(
                *pos, "x", color=colors[g], ms=10, label=f"{g.title()} goal"
            )
            self.path_lines[g],   = ax.plot([], [], "--",
                                            color=colors[g], lw=2, alpha=0.5)
            ox, oy = offsets[g]
            self.prob_texts[g] = ax.text(
                pos[0] + ox, pos[1] + oy, "0%",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                ha="center", va="center"
            )

        # ── Path-following state -------------------------------------------
        self.current_path: List[Tuple[float, float]] = []
        self.path_idx: int = 0
        self.target_goal = "tool_station"      # still follow tool-station path

    # ----------------------------------------------------------------------
    def plan_path(self):
        """Compute and cache Bézier-smoothed paths to every goal."""
        paths = {}
        for g, gpos in self.goals.items():
            raw = theta_star(self.human_pos, gpos, self.grid, self.pads)
            paths[g] = smooth_path_with_beziers(raw) if raw else [self.human_pos, gpos]
        return paths

    # ----------------------------------------------------------------------
    def update(self, _):
        """Matplotlib FuncAnimation callback """
        step_rem = ANIMATION_SPEED              # distance we must cover this frame

        # ── Re-plan if we have no path or just finished one ─────────────────
        if not self.current_path or self.path_idx >= len(self.current_path) - 1:
            self.current_path = self.plan_path()[self.target_goal]
            self.path_idx = 0
            self.human_pos = self.current_path[0]     # snap exactly onto new path

        # ── March forward along the poly-line until we've spent the step ───
        while step_rem > 0 and self.path_idx < len(self.current_path) - 1:
            target = self.current_path[self.path_idx + 1]
            dx, dy = target[0] - self.human_pos[0], target[1] - self.human_pos[1]
            seg_len = math.hypot(dx, dy)

            if seg_len <= step_rem:                   # we will *reach* the waypoint
                self.human_pos = target
                self.path_idx += 1
                step_rem -= seg_len
            else:                                     # advance part-way and stop
                ratio = step_rem / seg_len
                self.human_pos = (self.human_pos[0] + ratio * dx,
                                  self.human_pos[1] + ratio * dy)
                step_rem = 0

        # velocity for probability heuristic (direction only)
        self.human_vel = (dx, dy) if seg_len else (0, 0)
        self.human_history.append(self.human_pos)

        # marker
        self.human_marker.set_data([self.human_pos[0]], [self.human_pos[1]])

        # ── Calculate raw probabilities ───────────────────────────────────
        all_paths = self.plan_path()
        raw_probabilities: Dict[str, float] = {}

        for g, path in all_paths.items():
            if path:
                plen = sum(
                    math.hypot(path[i+1][0] - path[i][0],
                               path[i+1][1] - path[i][1])
                    for i in range(len(path)-1)
                )
                raw_probabilities[g] = calculate_path_probability(
                    self.human_pos, self.human_vel,
                    self.goals[g], g, plen, self.human_history
                )
            else:
                raw_probabilities[g] = 0.0

        # ── Temperature sharpening + normalisation ────────────────────────
        for g in raw_probabilities:
            raw_probabilities[g] **= TEMPERATURE
        total_prob = sum(raw_probabilities.values())

        if total_prob > 0:
            normalized_probabilities = {g: p / total_prob
                                         for g, p in raw_probabilities.items()}
        else:  # fallback – equally likely
            n = len(self.goals)
            normalized_probabilities = {g: 1.0 / n for g in self.goals}

        # ── Update visualisation ──────────────────────────────────────────
        for g, path in all_paths.items():
            if path:
                px, py = zip(*path)
                self.path_lines[g].set_data(px, py)
                self.prob_texts[g].set_text(f"{normalized_probabilities[g]:.1%}")

        # return the artists that changed
        return (self.human_marker,
                *self.path_lines.values(),
                *self.prob_texts.values())

# ----------------------------------------------------------------
#  top-level
# ----------------------------------------------------------------
def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    grid = Grid((X_MIN, X_MAX), (Y_MIN, Y_MAX), resolution=0.2)

    # workspace border
    ax.add_patch(
        patches.Rectangle(
            (X_MIN, Y_MIN), X_MAX - X_MIN, Y_MAX - Y_MIN,
            lw=2, ec="black", fc="none"
        )
    )

    # static obstacles ------------------------------------------------------
    pads, padded_objects = [], []
    desk    = create_padded(ax, (4.1 - 1.25, -3.5), 2.5, 7.0, "wheat",      "brown")
    battery = create_padded(ax, (-0.350, -0.850),   1.2, 0.5, "lightblue",  "blue")
    tool    = create_padded(ax, (-0.250,  1.200),   1.0, 0.5, "lightgreen", "green")
    agv     = create_padded(ax, AGV_POSITIONS[1],   1.75, 3.0,"lightcoral", "red")
    mobile  = create_padded(ax, MOBILE_POSITIONS[0],1.75, 0.891,"plum",     "purple")

    for obj in (desk, battery, tool, agv, mobile):
        padded_objects.append(obj)
        pads.append(obj["collision_data"])

    # animation ------------------------------------------------------------
    anim_obj = PathProbabilityAnimation(ax, grid, pads)
    anim = animation.FuncAnimation(fig, anim_obj.update, frames=FRAMES,
                                   interval=1_000 / FPS, blit=True, repeat=True)

    # ── annotate static objects ───────────────────────────────────────────
    static_labels = [
        ("Desk",           4.10,             0.00),
        ("Battery",        -0.35 + 0.60,    -0.85 + 0.25),
        ("Tool\nStation",  -0.25 + 0.50,     1.20 + 0.25),
        ("Workstation 1",  AGV_POSITIONS[1][0] + 0.875,
                           AGV_POSITIONS[1][1] + 1.50),
        ("Mobile Robot",   MOBILE_POSITIONS[0][0] + 0.875,
                           MOBILE_POSITIONS[0][1] + 0.4455),
    ]

    for txt, cx, cy in static_labels:
        ax.text(
            cx, cy, txt,
            ha="center", va="center",
            fontsize=9, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )

    # cosmetics ------------------------------------------------------------
    ax.set_xlim(X_MIN - 0.5, X_MAX + 0.5)
    ax.set_ylim(Y_MIN - 0.5, Y_MAX + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Human path planning with temperature-sharpened probabilities")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

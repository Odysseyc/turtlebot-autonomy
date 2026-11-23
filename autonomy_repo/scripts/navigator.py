#!/usr/bin/env python3
"""Navigator node: A* planning + cubic spline smoothing + differential-flatness tracking.

Implements the three functions BaseNavigator leaves empty (HW2 Problem 4):
  - compute_heading_control           (P control on heading, used for ALIGN / PARK)
  - compute_trajectory_tracking_control (feedback-linearizing tracking controller)
  - compute_trajectory_plan           (A* on the occupancy grid, then splrep smoothing)
"""

import heapq
import typing as T

import numpy as np
import rclpy
from scipy.interpolate import splev, splrep

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class AStar:
    """A* on an 8-connected grid with Euclidean edge costs (as in HW1).

    Nodes are integer grid indices (i, j); the matching state is (i * res, j * res).
    """

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal,
                 occupancy: StochOccupancyGrid2D, resolution: float) -> None:
        self.lo = np.asarray(statespace_lo, dtype=float)
        self.hi = np.asarray(statespace_hi, dtype=float)
        self.occupancy = occupancy
        self.res = resolution
        self.start = self.to_node(x_init)
        self.goal = self.to_node(x_goal)
        self.path: T.Optional[np.ndarray] = None

    def to_node(self, x) -> T.Tuple[int, int]:
        x = np.asarray(x, dtype=float)
        return (int(np.round(x[0] / self.res)), int(np.round(x[1] / self.res)))

    def to_state(self, node) -> np.ndarray:
        return np.array([node[0] * self.res, node[1] * self.res])

    def is_free(self, node) -> bool:
        x = self.to_state(node)
        if np.any(x < self.lo) or np.any(x > self.hi):
            return False
        return bool(self.occupancy.is_free(x))

    def heuristic(self, node) -> float:
        return float(np.linalg.norm(self.to_state(node) - self.to_state(self.goal)))

    def solve(self) -> bool:
        # The start is not collision-checked so a robot that drifted close to a
        # wall can still plan its way out; the goal must be free.
        if not self.is_free(self.goal):
            return False

        g = {self.start: 0.0}
        came_from = {}
        closed = set()
        counter = 0
        open_heap = [(self.heuristic(self.start), counter, self.start)]
        steps = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == self.goal:
                nodes = [current]
                while nodes[-1] in came_from:
                    nodes.append(came_from[nodes[-1]])
                self.path = np.array([self.to_state(n) for n in reversed(nodes)])
                return True
            closed.add(current)

            for di, dj in steps:
                neigh = (current[0] + di, current[1] + dj)
                if neigh in closed or not self.is_free(neigh):
                    continue
                tentative = g[current] + self.res * np.hypot(di, dj)
                if tentative < g.get(neigh, np.inf):
                    g[neigh] = tentative
                    came_from[neigh] = current
                    counter += 1
                    heapq.heappush(open_heap, (tentative + self.heuristic(neigh), counter, neigh))
        return False


class Navigator(BaseNavigator):
    def __init__(self) -> None:
        super().__init__("navigator")

        # heading controller gain
        self.declare_parameter("kp", 2.0)

        # tracking controller gains (same defaults as HW2 trajectory_tracking.py)
        self.declare_parameter("kpx", 2.0)
        self.declare_parameter("kpy", 2.0)
        self.declare_parameter("kdx", 2.0)
        self.declare_parameter("kdy", 2.0)

        # trajectory smoothing (same values as HW1 compute_smooth_plan)
        self.declare_parameter("v_desired", 0.15)
        self.declare_parameter("spline_alpha", 0.05)

        self.V_PREV_THRESH = 0.0001
        self.V_prev = 0.0
        self.om_prev = 0.0
        self.t_prev = 0.0

    # ----- ALIGN / PARK -----------------------------------------------------

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        err = wrap_angle(goal.theta - state.theta)
        return TurtleBotControl(omega=float(self.get_parameter("kp").value * err))

    # ----- TRACK ------------------------------------------------------------

    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan,
                                            t: float) -> TurtleBotControl:
        dt = t - self.t_prev

        x_d = float(splev(t, plan.path_x_spline, der=0))
        xd_d = float(splev(t, plan.path_x_spline, der=1))
        xdd_d = float(splev(t, plan.path_x_spline, der=2))
        y_d = float(splev(t, plan.path_y_spline, der=0))
        yd_d = float(splev(t, plan.path_y_spline, der=1))
        ydd_d = float(splev(t, plan.path_y_spline, der=2))

        # avoid the v = 0 singularity of the feedback-linearization Jacobian
        if abs(self.V_prev) < self.V_PREV_THRESH:
            self.V_prev = self.V_PREV_THRESH

        th = state.theta
        xd = self.V_prev * np.cos(th)
        yd = self.V_prev * np.sin(th)

        kpx = self.get_parameter("kpx").value
        kpy = self.get_parameter("kpy").value
        kdx = self.get_parameter("kdx").value
        kdy = self.get_parameter("kdy").value

        u = np.array([
            xdd_d + kpx * (x_d - state.x) + kdx * (xd_d - xd),
            ydd_d + kpy * (y_d - state.y) + kdy * (yd_d - yd),
        ])
        J = np.array([
            [np.cos(th), -self.V_prev * np.sin(th)],
            [np.sin(th), self.V_prev * np.cos(th)],
        ])
        a, om = np.linalg.solve(J, u)

        V = self.V_prev + a * dt

        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        # BaseController clips v and omega to the robot's limits
        return TurtleBotControl(v=float(V), omega=float(om))

    # ----- planning ---------------------------------------------------------

    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState,
                                occupancy: StochOccupancyGrid2D, resolution: float,
                                horizon: float) -> T.Optional[TrajectoryPlan]:
        astar = AStar(
            statespace_lo=(state.x - horizon, state.y - horizon),
            statespace_hi=(state.x + horizon, state.y + horizon),
            x_init=(state.x, state.y),
            x_goal=(goal.x, goal.y),
            occupancy=occupancy,
            resolution=resolution,
        )
        if not astar.solve() or len(astar.path) < 4:
            return None

        # reset the tracking controller history for the new plan
        self.V_prev = 0.0
        self.om_prev = 0.0
        self.t_prev = 0.0

        # time stamps from a constant-speed heuristic, then cubic spline fit
        path = astar.path
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        ts = np.concatenate([[0.0], np.cumsum(seg / self.get_parameter("v_desired").value)])
        alpha = self.get_parameter("spline_alpha").value
        path_x_spline = splrep(ts, path[:, 0], k=3, s=alpha)
        path_y_spline = splrep(ts, path[:, 1], k=3, s=alpha)

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=float(ts[-1]),
        )


if __name__ == "__main__":
    rclpy.init()
    node = Navigator()
    rclpy.spin(node)
    rclpy.shutdown()

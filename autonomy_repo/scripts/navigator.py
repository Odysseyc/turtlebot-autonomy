#!/usr/bin/env python3

import typing as T

import numpy as np
import rclpy
from scipy.interpolate import splev, splrep

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class AStar(object):
    """Represents a motion planning problem to be solved using A* (copied from HW1, plotting removed)"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1,
                 heuristic="l2", heuristic_weight=1.0):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a StochOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (m)
        self.heuristic_type = heuristic            # norm used by the heuristic: "l1", "l2", or "linf"
        self.heuristic_weight = heuristic_weight   # weight w >= 1 on the heuristic (w > 1 gives weighted A*)
        self.x_offset = x_init
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.heuristic(self.x_init)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        # check in bounds
        in_bounds = all(self.statespace_lo[i] <= x[i] <= self.statespace_hi[i] for i in range(len(x)))
        # check if colliding w obstacle or not (the ROS occupancy grid expects a numpy array)
        return in_bounds and self.occupancy.is_free(np.asarray(x))

    def distance(self, x1, x2):
        return np.sqrt(np.sum((np.array(x1)-np.array(x2))**2))

    def heuristic(self, x):
        # map heuristic type
        norm_orders = {"l1": 1, "l2": 2, "linf": np.inf}
        # raise error if needed
        if self.heuristic_type not in norm_orders:
            raise ValueError(f"Heuristic type: {self.heuristic_type}, choose l1, l2, or linf")
        diff = np.array(x) - np.array(self.x_goal)
        # w * ||x - x_goal||, where w = self.heuristic_weight
        return float(self.heuristic_weight * np.linalg.norm(diff, ord=norm_orders[self.heuristic_type]))

    def snap_to_grid(self, x):
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        neighbors = []
        r = self.resolution
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                # skip the state itself
                if dx == 0 and dy == 0:
                    continue
                # snap before checking so we test the exact state we return
                neighbor = self.snap_to_grid((x[0] + dx, x[1] + dy))
                if self.is_free(neighbor):
                    neighbors.append(neighbor)
        return neighbors

    def find_best_est_cost_through(self):
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        # still have nodes on the frontier to check
        while len(self.open_set) > 0:
            # find the lowest  cost one so far
            x_current = self.find_best_est_cost_through()
            # if it's the end, great. reconstruct path
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            # move from frontier to a node we've explored
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)

            # get the neighbors and check them
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue # we've already assessed this node
                # estimated cost of getting to neighbor from current node
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                # if neighbor hasn't been seen add it to open set
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                # otherwise, if the cost is higher this way from xcurrent, it's not optimal
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue

                # this is the best route to x_neigh found so far
                self.came_from[x_neigh] = x_current
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.heuristic(x_neigh)

        return False


class Navigator(BaseNavigator):
    def __init__(self) -> None:
        # let the base class set up the node, subscriptions, timers, and the
        # state machine that switches between heading / tracking / planning
        super().__init__("navigator")
        # proportional gain for heading control: how hard we turn per radian of error
        self.kp = 2.0

        # PD gains for the differential flatness trajectory tracker
        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0

        # tracker history: last commanded speed / turn rate and the time they were applied
        self.V_PREV_THRES = 0.0001
        self.V_prev = 0.0
        self.om_prev = 0.0
        self.t_prev = 0.0

        # planning / smoothing parameters (same defaults as compute_smooth_plan in HW1)
        self.v_desired = 0.15
        self.spline_alpha = 0.05

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        # how far off our heading is, wrapped to [-pi, pi] so we always turn the short way
        err = wrap_angle(goal.theta - state.theta)

        # proportional control: turn faster the further off we are
        omega = self.kp * err

        # package the angular velocity into a control message and send it back
        control = TurtleBotControl()
        control.omega = omega
        return control

    def compute_trajectory_tracking_control(
        self, state: TurtleBotState, plan: TrajectoryPlan, t: float
    ) -> TurtleBotControl:
        dt = t - self.t_prev

        # sample the desired position, velocity, and acceleration from the plan's splines
        x_d = splev(t, plan.path_x_spline, der=0)
        xd_d = splev(t, plan.path_x_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        y_d = splev(t, plan.path_y_spline, der=0)
        yd_d = splev(t, plan.path_y_spline, der=1)
        ydd_d = splev(t, plan.path_y_spline, der=2)

        # avoid singularity
        if abs(self.V_prev) < self.V_PREV_THRES:
            self.V_prev = self.V_PREV_THRES

        xd = self.V_prev * np.cos(state.theta)
        yd = self.V_prev * np.sin(state.theta)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx * (x_d - state.x) + self.kdx * (xd_d - xd),
                      ydd_d + self.kpy * (y_d - state.y) + self.kdy * (yd_d - yd)])

        # compute real controls
        J = np.array([[np.cos(state.theta), -self.V_prev * np.sin(state.theta)],
                      [np.sin(state.theta), self.V_prev * np.cos(state.theta)]])
        a, om = np.linalg.solve(J, u)
        V = self.V_prev + a * dt

        # save the commands that were applied and the time
        # (no clipping here: BaseNavigator clips the control before publishing)
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        control = TurtleBotControl()
        control.v = V
        control.omega = om
        return control

    def compute_trajectory_plan(
        self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
        # A* path planning in a square window of +/- horizon around the robot
        astar = AStar(
            (state.x - horizon, state.y - horizon),
            (state.x + horizon, state.y + horizon),
            (state.x, state.y),
            (goal.x, goal.y),
            occupancy,
            resolution=resolution,
        )
        if not astar.solve() or len(astar.path) < 4:
            return None

        # reset tracking controller history, since t restarts at 0 for the new plan
        self.V_prev = 0.0
        self.om_prev = 0.0
        self.t_prev = 0.0

        # time stamps for each waypoint assuming constant velocity between them
        path = np.asarray(astar.path)
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        ts = np.concatenate(([0.0], np.cumsum(segment_lengths / self.v_desired)))

        # cubic smoothing splines of x(t) and y(t)
        path_x_spline = splrep(ts, path[:, 0], k=3, s=self.spline_alpha)
        path_y_spline = splrep(ts, path[:, 1], k=3, s=self.spline_alpha)

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )


if __name__ == "__main__":
    # start ROS 2, create our navigator node, and keep it running until shut down
    rclpy.init()
    node = Navigator()
    rclpy.spin(node)
    rclpy.shutdown()

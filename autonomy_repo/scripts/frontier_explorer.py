#!/usr/bin/env python3
"""Frontier exploration node.

Picks the closest frontier from the latest /map and /state, sends it to the
navigator on /cmd_nav, and waits for /nav_success before choosing the next one.
Failed or timed-out goals are blacklisted so the robot keeps exploring instead
of stopping.
"""

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.signal import convolve2d

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState


# ---------------------------------------------------------------------------
# Ported from P1_frontier.py (unchanged)
# ---------------------------------------------------------------------------

def window_counts(mask, window_size):
    """Number of True cells of mask in the window_size x window_size window centered on every cell."""
    window = np.ones((window_size, window_size))
    return convolve2d(mask.astype(float), window, mode="same", boundary="fill", fillvalue=0)


def explore(occupancy, window_size=11, min_unknown=0.25, min_free=0.35, occupied_thresh=50):
    probs = occupancy.probs

    unknown = probs == -1
    occupied = probs >= occupied_thresh
    free = (probs >= 0) & (probs < occupied_thresh)

    unknown_counts = window_counts(unknown, window_size)
    occupied_counts = window_counts(occupied, window_size)
    free_counts = window_counts(free, window_size)

    total_cells = window_size ** 2
    frontier = (
        (unknown_counts >= min_unknown * total_cells)
        & (occupied_counts == 0)
        & (free_counts >= min_free * total_cells)
    )

    ky, kx = np.nonzero(frontier)
    return occupancy.grid2state(np.column_stack((kx, ky)))


def closest_frontier(frontier_states, current_state):
    if len(frontier_states) == 0:
        return None, np.inf
    distances = np.linalg.norm(frontier_states - current_state, axis=1)
    i = np.argmin(distances)
    return frontier_states[i], distances[i]


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class FrontierExplorer(Node):
    def __init__(self) -> None:
        super().__init__("frontier_explorer")

        # parameters
        self.declare_parameter("window_size", 11)         # frontier window (cells), as in Problem 1
        self.declare_parameter("decision_period", 1.0)    # [s] how often to check progress
        self.declare_parameter("goal_timeout", 60.0)      # [s] give up on a goal after this long
        self.declare_parameter("min_goal_dist", 0.3)      # [m] skip frontiers this close to the robot
        self.declare_parameter("blacklist_radius", 0.5)   # [m] skip frontiers near failed goals
        self.declare_parameter("stale_goal_dist", 0.3)    # [m] goal is stale if no frontier is this close
        self.declare_parameter("done_checks", 3)          # empty checks in a row before declaring done
        self.declare_parameter("stop_duration", 5.0)      # [s] how long to stop at a stop sign
        self.declare_parameter("ignore_duration", 5.0)    # [s] ignore detections after resuming

        # stop sign handling
        self.stopped = False
        self.resume_timer = None
        self.ignore_until = None   # rclpy Time; detections before this are ignored

        # latest data
        self.state = None        # TurtleBotState
        self.occupancy = None    # StochOccupancyGrid2D

        # exploration bookkeeping
        self.goal = None         # np.array([x, y]) of the goal currently sent to the navigator
        self.goal_time = None
        self.need_goal = True
        self.failed_goals = []
        self.empty_checks = 0
        self.done = False

        # interfaces with the navigator (see BaseNavigator / BaseController)
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.nav_success_callback, 10)
        self.detector_sub = self.create_subscription(Bool, "/detector_bool", self.detector_callback, 10)

        self.timer = self.create_timer(self.get_parameter("decision_period").value, self.decide)
        self.get_logger().info("Frontier explorer started")

    # ----- callbacks -------------------------------------------------------

    def state_callback(self, msg: TurtleBotState) -> None:
        self.state = msg

    def map_callback(self, msg: OccupancyGrid) -> None:
        # same construction as BaseNavigator.map_callback
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

    def detector_callback(self, msg: Bool) -> None:
        """Stop for stop_duration seconds when a stop sign is detected."""
        if not msg.data or self.done or self.stopped or self.state is None:
            return

        # right after resuming, the same stop sign may still be in view: ignore it
        now = self.get_clock().now()
        if self.ignore_until is not None and now < self.ignore_until:
            return

        self.stopped = True
        stop_duration = self.get_parameter("stop_duration").value
        self.get_logger().info(f"Stop sign detected: stopping for {stop_duration:.1f} s")

        # The navigator owns /cmd_vel, so stop it the navigator's way: command the
        # current pose as the goal. It is already "near" and aligned, so the
        # navigator parks immediately and goes IDLE (zero velocity).
        self.cmd_nav_pub.publish(
            TurtleBotState(x=self.state.x, y=self.state.y, theta=self.state.theta)
        )

        # one-shot timer to resume exploration
        self.resume_timer = self.create_timer(stop_duration, self.resume_exploration)

    def resume_exploration(self) -> None:
        # make the timer one-shot
        self.resume_timer.cancel()
        self.destroy_timer(self.resume_timer)
        self.resume_timer = None

        self.stopped = False
        ignore_duration = self.get_parameter("ignore_duration").value
        self.ignore_until = self.get_clock().now() + Duration(seconds=ignore_duration)
        self.get_logger().info(
            f"Resuming exploration; ignoring stop signs for {ignore_duration:.1f} s"
        )

        # pick the closest frontier again (usually the same goal as before the stop)
        self.goal = None
        self.need_goal = True
        self.decide()

    def nav_success_callback(self, msg: Bool) -> None:
        if self.done or self.stopped or self.goal is None:
            return
        if msg.data:
            self.get_logger().info(f"Reached frontier goal ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        else:
            # e.g. the goal sits next to a wall that was just mapped (Problem 1 (iii))
            self.get_logger().warn(
                f"Planning failed for goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}); picking another"
            )
            self.failed_goals.append(self.goal.copy())
        self.goal = None
        self.need_goal = True

    # ----- decision logic --------------------------------------------------

    def decide(self) -> None:
        if self.done or self.stopped or self.state is None or self.occupancy is None:
            return

        frontiers = explore(self.occupancy, window_size=self.get_parameter("window_size").value)
        candidates = self.filter_candidates(frontiers)

        # still driving to a goal: check for timeout or a goal that is already mapped
        if self.goal is not None and not self.need_goal:
            elapsed = (self.get_clock().now() - self.goal_time).nanoseconds * 1e-9
            if elapsed > self.get_parameter("goal_timeout").value:
                self.get_logger().warn("Goal timed out; blacklisting it and picking another")
                self.failed_goals.append(self.goal.copy())
                self.need_goal = True
            elif self.goal_is_stale(frontiers):
                self.get_logger().info("Current goal already mapped; picking a new frontier")
                self.need_goal = True
            else:
                return

        if len(candidates) == 0:
            self.empty_checks += 1
            if self.empty_checks >= self.get_parameter("done_checks").value:
                self.done = True
                self.goal = None
                self.get_logger().info("No frontiers left: exploration complete")
            return

        self.empty_checks = 0
        self.send_goal(candidates)

    def filter_candidates(self, frontiers: np.ndarray) -> np.ndarray:
        """Drop frontiers too close to the robot (the navigator would not report success)
        and frontiers near goals that already failed."""
        if len(frontiers) == 0:
            return frontiers
        pos = np.array([self.state.x, self.state.y])
        keep = np.linalg.norm(frontiers - pos, axis=1) >= self.get_parameter("min_goal_dist").value
        radius = self.get_parameter("blacklist_radius").value
        for g in self.failed_goals:
            keep &= np.linalg.norm(frontiers - g, axis=1) >= radius
        return frontiers[keep]

    def goal_is_stale(self, frontiers: np.ndarray) -> bool:
        if len(frontiers) == 0:
            return True
        _, dist = closest_frontier(frontiers, self.goal)
        return dist > self.get_parameter("stale_goal_dist").value

    def send_goal(self, candidates: np.ndarray) -> None:
        pos = np.array([self.state.x, self.state.y])
        goal_xy, dist = closest_frontier(candidates, pos)

        # face the direction of travel so the navigator does not spin on arrival
        theta = float(np.arctan2(goal_xy[1] - pos[1], goal_xy[0] - pos[0]))

        self.goal = np.array(goal_xy, dtype=float)
        self.goal_time = self.get_clock().now()
        self.need_goal = False
        self.cmd_nav_pub.publish(TurtleBotState(x=float(goal_xy[0]), y=float(goal_xy[1]), theta=theta))
        self.get_logger().info(
            f"New frontier goal ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}), {dist:.2f} m away, "
            f"{len(candidates)} candidates"
        )


def main() -> None:
    rclpy.init()
    node = FrontierExplorer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

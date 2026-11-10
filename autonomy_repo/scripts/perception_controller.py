#!/usr/bin/env python3

import numpy as np
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from std_msgs.msg import Bool


class PerceptionController(BaseHeadingController):
    def __init__(self) -> None:
        # let the base class set up the node, subscriptions, and control loop
        super().__init__("perception_controller")
        # proportional gain: how hard we turn per radian of heading error
        self.declare_parameter("kp", 2.0)
        self.declare_parameter("active", True)
        self.image_detected = False
        self.image_detector = self.create_subscription(Bool, "/detector_bool", self.detector_callback, 10)

    def detector_callback(self, msg: Bool) -> None:
        if msg.data == True:
            self.image_detected = True
        else:
            self.image_detected = False
        
    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value

    @property
    def active(self) -> bool:
        return self.get_parameter("active").value


    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        # how far off our heading is, wrapped to [-pi, pi] so we always turn the short way
        err = wrap_angle(goal.theta - state.theta)

        # proportional control: turn faster the further off we are
        if self.image_detected == False:
            omega = 0.2
        else:
            omega = 0.0

        # package the angular velocity into a control message and send it back
        control = TurtleBotControl()
        control.omega = omega
        return control


if __name__ == "__main__":
    # start ROS 2, create our controller node, and keep it running until shut down
    rclpy.init()
    node = PerceptionController()
    rclpy.spin(node)
    rclpy.shutdown()

#!/usr/bin/env python3

from rcl_interfaces import msg
import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import Int64, Bool, String
from geometry_msgs.msg import Twist


class Heartbeat(Node):
    def __init__(self) -> None:
# initialize base class (must happen before everything else)
        super().__init__("heartbeat")

        # create publisher with: self.create_publisher(<msg type>, <topic>, <qos>)
        self.hb_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # create a timer with: self.create_timer(<second>, <callback>)
        self.hb_timer = self.create_timer(0.2, self.hb_callback)

        # create a subscriber for the kill topic
        self.kill_sub = self.create_subscription(Bool, "/kill", self.kill_callback, 10)

    def hb_callback(self) -> None:
        """
        Heartbeat callback triggered by the timer
        """
       # construct heartbeat message
        msg = Twist()
        msg.linear.x = 1.0  # set this to be the linear velocity
        msg.angular.z = 1.0 # set this to be the angular veloT, "/ccity


        self.hb_pub.publish(msg)

    def kill_callback(self, msg: Bool) -> None:
        """
        Kill callback triggered by the kill topic
        """
        if msg.data:
            # stop the timer
            self.hb_timer.cancel()

            # publish a zero velocity message to stop the robot
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.linear.y = 0.0
            stop_msg.linear.z = 0.0
            stop_msg.angular.x = 0.0
            stop_msg.angular.y = 0.0
            stop_msg.angular.z = 0.0

            self.get_logger().info("Kill received, sending zero cmd_vel")
            self.hb_pub.publish(stop_msg)
        
            

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = Heartbeat()  # instantiate the heartbeat node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 contextT, "/c

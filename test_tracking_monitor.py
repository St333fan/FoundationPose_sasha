#!/usr/bin/env python3
"""
Test client to monitor FoundationPose tracking output.
Subscribes to tracked object poses and prints them.
"""

import rospy
from geometry_msgs.msg import PoseStamped
import sys

class TrackingMonitor:
    def __init__(self, object_names):
        rospy.init_node('foundationpose_tracking_monitor')
        
        self.subscribers = {}
        self.latest_poses = {}
        
        for obj_name in object_names:
            topic = f'/pose_estimator/foundationpose/{obj_name}/pose'
            self.subscribers[obj_name] = rospy.Subscriber(
                topic, 
                PoseStamped, 
                lambda msg, name=obj_name: self.pose_callback(msg, name)
            )
            self.latest_poses[obj_name] = None
            rospy.loginfo(f"Subscribed to {topic}")
        
        # Timer to print poses periodically
        self.print_timer = rospy.Timer(rospy.Duration(1.0), self.print_poses)
        
    def pose_callback(self, msg, obj_name):
        """Store latest pose for each object"""
        self.latest_poses[obj_name] = msg
        
    def print_poses(self, event):
        """Print all tracked object poses"""
        print("\n" + "="*80)
        print(f"Tracked Object Poses (Time: {rospy.Time.now().to_sec():.2f})")
        print("="*80)
        
        for obj_name, pose_msg in self.latest_poses.items():
            if pose_msg is not None:
                pos = pose_msg.pose.position
                ori = pose_msg.pose.orientation
                age = (rospy.Time.now() - pose_msg.header.stamp).to_sec()
                
                print(f"\n{obj_name}:")
                print(f"  Position: [{pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}]")
                print(f"  Orientation: [w:{ori.w:.4f}, x:{ori.x:.4f}, y:{ori.y:.4f}, z:{ori.z:.4f}]")
                print(f"  Age: {age:.3f}s")
            else:
                print(f"\n{obj_name}: No data received yet")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_tracking_monitor.py <object_name1> [object_name2] ...")
        print("Example: python test_tracking_monitor.py 001_ahorn_sirup 002_max_house")
        sys.exit(1)
    
    object_names = sys.argv[1:]
    print(f"Monitoring tracking for objects: {object_names}")
    
    monitor = TrackingMonitor(object_names)
    rospy.spin()

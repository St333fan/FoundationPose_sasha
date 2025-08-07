import os
import sys
import json
import time
import argparse
import numpy as np
import trimesh
import nvdiffrast.torch as dr

# Add current directory to path for imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)

from estimater import FoundationPose
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
from Utils import set_logging_format, set_seed

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
import actionlib
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
import ros_numpy
import transforms3d as tf3d
import cv2


class FoundationPose_ROS:
    def __init__(self, config_file):
        print(f"Using config file: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.object_name_mapping = config["object_mapping"]
        self.intrinsics = np.asarray(config['cam_K']).reshape((3, 3))
        print(f"Using intrinsics: {self.intrinsics}")

        # Configuration parameters
        self.est_refine_iter = config.get('est_refine_iter', 5)

        # Set up logging and random seed
        set_logging_format()
        set_seed(0)

        rospy.loginfo("Initializing FoundationPose components")
        
        # Initialize shared FoundationPose components (these are the memory-heavy parts)
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        
        # Store mesh information for each object
        self.mesh_files = config.get('mesh_files', {})
        self.meshes = {}  # Store loaded meshes
        self.current_estimator = None  # Single estimator instance
        self.current_object = None  # Track which object the estimator is configured for
        
        # Pre-load meshes
        for obj_name, mesh_file in self.mesh_files.items():
            if os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file)
                self.meshes[obj_name] = mesh
                rospy.loginfo(f"Loaded mesh for {obj_name}: {mesh_file}")
            else:
                rospy.logwarn(f"Mesh file not found for {obj_name}: {mesh_file}")

        # Initialize ROS node and action server
        rospy.init_node("foundationpose_estimation")
        self.server = actionlib.SimpleActionServer('/pose_estimator/foundationpose',
                                                   GenericImgProcAnnotatorAction,
                                                   execute_cb=self.estimate_pose,
                                                   auto_start=False)
        self.server.start()
        print("Pose Estimation with FoundationPose is ready.")

    def get_estimator_for_object(self, obj_name):
        """Get or create estimator for specific object (memory efficient)"""
        if self.current_object != obj_name or self.current_estimator is None:
            # Need to switch to different object or create first estimator
            rospy.loginfo(f"Switching estimator to object: {obj_name}")
            
            mesh = self.meshes[obj_name]
            self.current_estimator = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=self.scorer,  # Shared
                refiner=self.refiner,  # Shared
                debug_dir=None,
                debug=0,
                glctx=self.glctx  # Shared
            )
            self.current_object = obj_name
                
        return self.current_estimator

    def estimate_pose(self, req):
        print("Request detection...")
        start_time = time.time()

        # Extract data from request
        mask_detections = req.mask_detections
        class_names = req.class_names
        rgb = req.rgb
        depth = req.depth

        # Print incoming class names for debugging
        print(f"Received class names: {class_names}")
        print(f"Number of detected objects: {len(class_names)}")
        
        for i, name in enumerate(class_names):
            print(f"  Object {i+1}: {name}")
            if name in self.object_name_mapping:
                mapped_name = self.object_name_mapping[name]
                print(f"    -> Mapped to: {mapped_name}")
                if mapped_name in self.meshes:
                    print(f"    -> Mesh available: YES")
                else:
                    print(f"    -> Mesh available: NO")
            else:
                print(f"    -> Not found in object mapping")

        width, height = rgb.width, rgb.height
        print(f"Image dimensions: {width}x{height}")
        
        # Convert ROS messages to numpy arrays
        image = ros_numpy.numpify(rgb)
        
        try:
            depth_img = ros_numpy.numpify(depth)
            print("Depth image: Available")
        except Exception as e:
            rospy.logwarn("Missing depth image in the goal.")
            depth_img = None
            print("Depth image: Not available")

        print("RGB", image.shape, image.dtype)
        
        if depth_img is not None:
            print("Depth", depth_img.shape, depth_img.dtype)

        # Convert mask detections
        mask_detections = [ros_numpy.numpify(mask_img).astype(np.uint8)
                          for mask_img in req.mask_detections]

        if mask_detections:
            print("Mask", mask_detections[0].shape, mask_detections[0].dtype)
            print(f"Number of masks: {len(mask_detections)}")

        # Process each detected object
        valid_class_names = []
        pose_results = []

        for i, class_name in enumerate(class_names):
            print(f"Processing object {i+1}/{len(class_names)}: {class_name}")
            
            # Check if we have a mapping for this class
            if class_name not in self.object_name_mapping:
                print(f"  -> No mapping found for {class_name}, skipping")
                continue
                
            mapped_name = self.object_name_mapping[class_name]
            print(f"  -> Mapped to: {mapped_name}")
            
            # Check if we have mesh for this object
            if mapped_name not in self.meshes:
                print(f"  -> No mesh available for {mapped_name}, skipping")
                continue
                
            # Check if we have corresponding mask
            if i >= len(mask_detections):
                print(f"  -> No mask available for object {i}, skipping")
                continue
                
            mask = mask_detections[i]
            
            # Get estimator for this object
            try:
                estimator = self.get_estimator_for_object(mapped_name)
                print(f"  -> Estimator ready for {mapped_name}")
                
                # Run pose estimation (no tracking, fresh estimation each time)
                pose_result = estimator.register(K=self.intrinsics, rgb=image, depth=depth_img, 
                                                ob_mask=mask, iteration=self.est_refine_iter)
                
                if pose_result is not None:
                    # Convert pose to ROS Pose message
                    pose_msg = Pose()
                    
                    # Extract translation and rotation from pose_result
                    if hasattr(pose_result, 'shape') and pose_result.shape == (4, 4):
                        # pose_result is a 4x4 transformation matrix
                        translation = pose_result[:3, 3]
                        rotation_matrix = pose_result[:3, :3]
                        
                        # Convert rotation matrix to quaternion
                        quaternion = tf3d.quaternions.mat2quat(rotation_matrix)
                        
                        pose_msg.position = Point(x=float(translation[0]), 
                                                y=float(translation[1]), 
                                                z=float(translation[2]))
                        pose_msg.orientation = Quaternion(x=float(quaternion[1]), 
                                                        y=float(quaternion[2]), 
                                                        z=float(quaternion[3]), 
                                                        w=float(quaternion[0]))
                        
                        pose_results.append(pose_msg)
                        valid_class_names.append(class_name)
                        print(f"  -> Pose estimation successful for {class_name}")
                    else:
                        print(f"  -> Invalid pose result format for {class_name}")
                else:
                    print(f"  -> Pose estimation failed for {class_name}")
                    
            except Exception as e:
                print(f"  -> Error processing {class_name}: {str(e)}")
                rospy.logwarn(f"Error processing {class_name}: {str(e)}")

        # Simple response for testing
        response = GenericImgProcAnnotatorResult()
        response.pose_results = pose_results
        response.class_names = valid_class_names

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        
        self.server.set_succeeded(response)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', 
                       default="./foundationpose_configs/cfg_ros_ycbv_inference.json",
                       help='Path to configuration file')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    foundationpose_ros = FoundationPose_ROS(**vars(opt))
    
    rospy.spin()

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
            
            if obj_name not in self.meshes:
                raise ValueError(f"No mesh available for object: {obj_name}")
            
            mesh = self.meshes[obj_name]
            print(f"  -> Using mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            # Ensure vertex normals are available
            if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
                rospy.loginfo(f"Computing vertex normals for {obj_name}")
                mesh.compute_vertex_normals()
            
            print(f"  -> Mesh vertex normals shape: {mesh.vertex_normals.shape if mesh.vertex_normals is not None else 'None'}")
            
            try:
                self.current_estimator = FoundationPose(
                    model_pts=mesh.vertices,
                    model_normals=mesh.vertex_normals,
                    mesh=mesh,
                    scorer=self.scorer,  # Shared
                    refiner=self.refiner,  # Shared
                    debug_dir="debug",  # this is needed for FoundationPose, but we don't use it here
                    debug=0,
                    glctx=self.glctx  # Shared
                )
                self.current_object = obj_name
                print(f"  -> FoundationPose estimator created successfully for {obj_name}")
            except Exception as e:
                print(f"  -> ERROR creating FoundationPose estimator: {str(e)}")
                raise e
                
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
        
        # Ensure RGB image is in correct format (H, W, 3) with uint8
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                if image.max() <= 1.0:  # Normalized to [0,1]
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
        else:
            print(f"WARNING: Unexpected RGB image format: {image.shape}, dtype: {image.dtype}")
        
        try:
            depth_img = ros_numpy.numpify(depth)
            # Convert depth image to supported dtype (float32) if needed
            if depth_img.dtype == np.uint16:
                # Convert from millimeters to meters if needed
                depth_img = depth_img.astype(np.float32) / 1000.0
            elif depth_img.dtype != np.float32:
                depth_img = depth_img.astype(np.float32)
            print("Depth image: Available")
            print(f"Depth range: min={depth_img.min():.3f}, max={depth_img.max():.3f}")
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
            print(f"First mask unique values: {np.unique(mask_detections[0])}")
            
            # Ensure masks are binary (0 or 255)
            for i, mask in enumerate(mask_detections):
                if mask.max() > 1:
                    # Already in 0-255 format
                    pass
                else:
                    # Convert from 0-1 to 0-255
                    mask_detections[i] = (mask * 255).astype(np.uint8)
                    print(f"Converted mask {i} from 0-1 to 0-255 format")

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
            
            # Debug mask information
            mask_pixels = np.sum(mask > 0)
            print(f"  -> Mask has {mask_pixels} non-zero pixels")
            if mask_pixels == 0:
                print(f"  -> Empty mask for {class_name}, skipping")
                continue
            
            # Get estimator for this object
            try:
                estimator = self.get_estimator_for_object(mapped_name)
                print(f"  -> Estimator ready for {mapped_name}")
                
                # Debug input data
                print(f"  -> Image shape: {image.shape}, depth shape: {depth_img.shape if depth_img is not None else 'None'}")
                print(f"  -> Mask shape: {mask.shape}, mask dtype: {mask.dtype}")
                print(f"  -> Intrinsics: {self.intrinsics}")
                print(f"  -> Refinement iterations: {self.est_refine_iter}")
                
                # Run pose estimation (no tracking, fresh estimation each time)
                print(f"  -> Running pose estimation for {mapped_name}...")
                
                # Check if depth is available - FoundationPose needs depth for registration
                if depth_img is None:
                    print(f"  -> ERROR: Depth image required for pose estimation but not available")
                    continue
                
                pose_result = estimator.register(K=self.intrinsics, rgb=image, depth=depth_img, 
                                                ob_mask=mask, iteration=self.est_refine_iter)
                
                print(f"  -> Pose estimation returned: {type(pose_result)}")
                if pose_result is not None:
                    print(f"  -> Pose result shape: {pose_result.shape if hasattr(pose_result, 'shape') else 'No shape'}")
                    print(f"  -> Pose result:\n{pose_result}")
                    
                    # Convert pose to ROS Pose message
                    pose_msg = Pose()
                    
                    # Extract translation and rotation from pose_result
                    if hasattr(pose_result, 'shape') and pose_result.shape == (4, 4):
                        # pose_result is a 4x4 transformation matrix
                        translation = pose_result[:3, 3]
                        rotation_matrix = pose_result[:3, :3]
                        
                        print(f"  -> Translation: {translation}")
                        print(f"  -> Rotation matrix:\n{rotation_matrix}")
                        
                        # Check if pose is identity/zero (indicating failure)
                        if np.allclose(translation, 0) and np.allclose(rotation_matrix, np.eye(3)):
                            print(f"  -> WARNING: Pose result is identity matrix - estimation likely failed")
                        
                        # Convert rotation matrix to quaternion
                        quaternion = tf3d.quaternions.mat2quat(rotation_matrix)
                        print(f"  -> Quaternion: {quaternion}")
                        
                        pose_msg.position = Point(x=float(translation[0]), 
                                                y=float(translation[1]), 
                                                z=float(translation[2]))
                        pose_msg.orientation = Quaternion(x=float(quaternion[1]), 
                                                        y=float(quaternion[2]), 
                                                        z=float(quaternion[3]), 
                                                        w=float(quaternion[0]))
                        
                        print(f"  -> Finished pose for {class_name}:")
                        print(f"     Position: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]")
                        print(f"     Quaternion: [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]")
                        
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

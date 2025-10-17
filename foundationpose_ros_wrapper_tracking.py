import os
import sys
import json
import time
import argparse
import numpy as np
import trimesh
import nvdiffrast.torch as dr
from estimater import FoundationPose
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
from Utils import set_logging_format, set_seed
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from sensor_msgs.msg import Image
import actionlib
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
import ros_numpy
import transforms3d as tf3d
import cv2
from threading import Lock
import message_filters

### Some implementation strategies are from https://github.com/ammar-n-abbas/FoundationPoseROS2 ###

# Add current directory to path for imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)

# Patch FoundationPose to add is_register flag (like in ROS2 version)
original_register = FoundationPose.register

def patched_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True  # Set flag after successful registration
    return pose

FoundationPose.register = patched_register

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
        self.track_refine_iter = config.get('track_refine_iter', 2)

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
        
        # Tracking state management (memory efficient - one estimator per tracked object)
        self.tracked_objects = {}  # {obj_name: {'estimator': FoundationPose, 'pose': np.array, 'publisher': rospy.Publisher}}
        self.tracking_lock = Lock()  # Thread safety for tracking state
        
        # Latest sensor data for tracking
        self.latest_rgb = None
        self.latest_depth = None
        self.sensor_data_lock = Lock()
        
        # Pre-load meshes
        for obj_name, mesh_file in self.mesh_files.items():
            if os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file)
                
                # Scale mesh from millimeters to meters (YCB-V models are in mm)
                # FoundationPose expects models in meters to match camera depth units
                rospy.loginfo(f"Scaling mesh {obj_name} from mm to meters (factor: 0.001)")
                mesh.vertices = mesh.vertices / 1000.0  # Convert mm to m
                
                self.meshes[obj_name] = mesh
                rospy.loginfo(f"Loaded and scaled mesh for {obj_name}: {mesh_file}")
            else:
                rospy.logwarn(f"Mesh file not found for {obj_name}: {mesh_file}")

        # Initialize ROS node and action server
        rospy.init_node("foundationpose_estimation")
        
        # Subscribe to camera topics for continuous tracking
        rgb_topic = config.get('rgb_topic', '/hsrb/head_rgbd_sensor/rgb/image_rect_color')
        depth_topic = config.get('depth_topic', '/hsrb/head_rgbd_sensor/depth_registered/image_raw')
        
        rospy.loginfo(f"Subscribing to RGB topic: {rgb_topic}")
        rospy.loginfo(f"Subscribing to Depth topic: {depth_topic}")
        
        # Use message_filters for synchronized RGB-D
        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        
        # Synchronize RGB and Depth messages
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.sensor_callback)
        
        self.server = actionlib.SimpleActionServer('/pose_estimator/foundationpose',
                                                   GenericImgProcAnnotatorAction,
                                                   execute_cb=self.estimate_pose,
                                                   auto_start=False)
        self.server.start()
        
        # Start tracking update timer (runs at ~5Hz)
        self.tracking_timer = rospy.Timer(rospy.Duration(1/20), self.tracking_update)
        
        print("\n" + "="*80)
        print("Pose Estimation with FoundationPose is ready.")
        print("Tracking enabled - will publish poses continuously for registered objects.")
        print(f"Tracking timer running at {1/0.2:.1f} Hz")
        print(f"RGB topic: {rgb_topic}")
        print(f"Depth topic: {depth_topic}")
        print(f"Loaded meshes: {list(self.meshes.keys())}")
        print("="*80 + "\n")

    def sensor_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB-D sensor data"""
        with self.sensor_data_lock:
            try:
                # Convert ROS messages to numpy arrays
                self.latest_rgb = ros_numpy.numpify(rgb_msg)
                
                # Ensure RGB image is in correct format
                if len(self.latest_rgb.shape) == 3 and self.latest_rgb.shape[2] == 3:
                    if self.latest_rgb.dtype != np.uint8:
                        if self.latest_rgb.max() <= 1.0:
                            self.latest_rgb = (self.latest_rgb * 255).astype(np.uint8)
                        else:
                            self.latest_rgb = self.latest_rgb.astype(np.uint8)
                
                # Convert depth
                self.latest_depth = ros_numpy.numpify(depth_msg)
                if self.latest_depth.dtype == np.uint16:
                    self.latest_depth = self.latest_depth.astype(np.float32) / 1000.0
                elif self.latest_depth.dtype != np.float32:
                    self.latest_depth = self.latest_depth.astype(np.float32)
                    
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"Error in sensor callback: {str(e)}")

    def tracking_update(self, event):
        """Periodic tracking update for all registered objects"""
        with self.sensor_data_lock:
            rgb = self.latest_rgb
            depth = self.latest_depth
        
        # No sensor data yet
        if rgb is None or depth is None:
            return
        
        with self.tracking_lock:
            num_tracked = len(self.tracked_objects)
            if num_tracked == 0:
                return  # Nothing to track
                
            # Track each registered object
            for obj_name, obj_data in list(self.tracked_objects.items()):
                try:
                    estimator = obj_data['estimator']
                    
                    # Check if registration was completed (like ROS2 version)
                    if not hasattr(estimator, 'is_register') or not estimator.is_register:
                        rospy.logwarn_throttle(5.0, f"Estimator for {obj_name} not registered yet, skipping")
                        continue
                    
                    # Check if estimator has pose_last set (required for tracking)
                    if not hasattr(estimator, 'pose_last') or estimator.pose_last is None:
                        rospy.logwarn_throttle(5.0, f"Estimator for {obj_name} doesn't have pose_last set, skipping")
                        continue
                    
                    # Run tracking (uses the last known pose as initialization)
                    pose = estimator.track_one(
                        rgb=rgb, 
                        depth=depth, 
                        K=self.intrinsics, 
                        iteration=self.track_refine_iter
                    )
                    
                    if pose is not None:
                        # Update stored pose
                        obj_data['pose'] = pose
                        
                        # Publish pose
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = rospy.Time.now()
                        pose_msg.header.frame_id = "head_rgbd_sensor_rgb_frame"
                        
                        translation = pose[:3, 3]
                        rotation_matrix = pose[:3, :3]
                        quaternion = tf3d.quaternions.mat2quat(rotation_matrix)
                        
                        pose_msg.pose.position = Point(
                            x=float(translation[0]),
                            y=float(translation[1]),
                            z=float(translation[2])
                        )
                        pose_msg.pose.orientation = Quaternion(
                            x=float(quaternion[1]),
                            y=float(quaternion[2]),
                            z=float(quaternion[3]),
                            w=float(quaternion[0])
                        )
                        
                        obj_data['publisher'].publish(pose_msg)
                        rospy.loginfo_throttle(2.0, f"Tracking {obj_name}: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
                        
                except Exception as e:
                    rospy.logwarn_throttle(5.0, f"Tracking error for {obj_name}: {str(e)}")

    def create_estimator_for_object(self, obj_name):
        """Create a new estimator instance for an object"""
        if obj_name not in self.meshes:
            raise ValueError(f"No mesh available for object: {obj_name}")
        
        mesh = self.meshes[obj_name]
        rospy.loginfo(f"Creating estimator for {obj_name}")
        print(f"  -> Using mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Ensure vertex normals are available
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            rospy.loginfo(f"Computing vertex normals for {obj_name}")
            mesh.compute_vertex_normals()
        
        estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,  # Shared
            refiner=self.refiner,  # Shared
            debug_dir="debug",
            debug=0,
            glctx=self.glctx  # Shared
        )
        
        print(f"  -> FoundationPose estimator created successfully for {obj_name}")
        return estimator

    def start_tracking_object(self, obj_name, initial_pose, estimator):
        """Start tracking an object with given initial pose and estimator instance"""
        with self.tracking_lock:
            # Check if already tracking - stop and restart with new estimator
            if obj_name in self.tracked_objects:
                rospy.loginfo(f"Object {obj_name} is already being tracked, restarting with new pose")
                # Unregister old publisher
                self.tracked_objects[obj_name]['publisher'].unregister()
            
            # Use the provided estimator (which has pose_last set from registration)
            try:
                # Create publisher for this object
                topic_name = f'/pose_estimator/foundationpose/{obj_name}/pose'
                publisher = rospy.Publisher(topic_name, PoseStamped, queue_size=10)
                
                # Register for tracking
                self.tracked_objects[obj_name] = {
                    'estimator': estimator,  # Reuse the estimator that did registration
                    'pose': initial_pose,
                    'publisher': publisher
                }
                
                rospy.loginfo(f"Started tracking {obj_name}, publishing to {topic_name}")
                
            except Exception as e:
                rospy.logerr(f"Failed to start tracking {obj_name}: {str(e)}")

    def stop_tracking_object(self, obj_name):
        """Stop tracking an object and free resources"""
        with self.tracking_lock:
            if obj_name in self.tracked_objects:
                # Unregister publisher
                self.tracked_objects[obj_name]['publisher'].unregister()
                # Remove from tracked objects (estimator will be garbage collected)
                del self.tracked_objects[obj_name]
                rospy.loginfo(f"Stopped tracking {obj_name}")

    def get_estimator_for_object(self, obj_name):
        """Get or create estimator for specific object (for initial registration)"""
        # Always create a new estimator for registration
        # This ensures clean state and allows us to track the same estimator afterward
        return self.create_estimator_for_object(obj_name)

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
                print(f"  -> Estimator.is_register after registration: {getattr(estimator, 'is_register', 'NOT SET')}")
                print(f"  -> Estimator.pose_last after registration: {'SET' if (hasattr(estimator, 'pose_last') and estimator.pose_last is not None) else 'NOT SET'}")
                
                if pose_result is not None:
                    
                    # Convert pose to ROS Pose message
                    pose_msg = Pose()
                    
                    # Extract translation and rotation from pose_result
                    if hasattr(pose_result, 'shape') and pose_result.shape == (4, 4):
                        # pose_result is a 4x4 transformation matrix
                        translation = pose_result[:3, 3]
                        rotation_matrix = pose_result[:3, :3]

                        # Check if pose is identity/zero (indicating failure)
                        if np.allclose(translation, 0) and np.allclose(rotation_matrix, np.eye(3)):
                            print(f"  -> WARNING: Pose result is identity matrix - estimation likely failed")
                        
                        # Convert rotation matrix to quaternion
                        quaternion = tf3d.quaternions.mat2quat(rotation_matrix)

                        pose_msg.position = Point(x=float(translation[0]), 
                                                y=float(translation[1]), 
                                                z=float(translation[2]))
                        pose_msg.orientation = Quaternion(x=float(quaternion[1]), 
                                                        y=float(quaternion[2]), 
                                                        z=float(quaternion[3]), 
                                                        w=float(quaternion[0]))
                        
                        print(f"  -> Finished pose for {class_name}:")
                        print(f"     Position: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]")
                        
                        pose_results.append(pose_msg)
                        valid_class_names.append(class_name)
                        print(f"  -> Pose estimation successful for {class_name}")
                        
                        # Start tracking this object with the initial pose and the estimator
                        # IMPORTANT: Pass the estimator that did the registration so pose_last is preserved
                        self.start_tracking_object(mapped_name, pose_result, estimator)
                        print(f"  -> Started tracking for {mapped_name}")
                        
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
                       default="./foundationpose_configs/cfg_ros_manibot_inference.json",
                       help='Path to configuration file')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    foundationpose_ros = FoundationPose_ROS(**vars(opt))
    
    rospy.spin()

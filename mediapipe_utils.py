import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import traceback
import config

def initialize_mediapipe_models(model_file_name):
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_model_file_path = os.path.join(script_dir, "models", model_file_name)
    
    pose_landmarker_obj = None
    if not os.path.exists(absolute_model_file_path):
        print(f"Error: Pose landmarker model not found at {absolute_model_file_path}")
        return None, None
    else:
        try:
            with open(absolute_model_file_path, 'rb') as f: model_asset_buffer = f.read()
            BaseOptions, PoseLandmarkerOptions, VisionRunningMode = mp.tasks.BaseOptions, vision.PoseLandmarkerOptions, vision.RunningMode
            options_pose = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model_asset_buffer),
                running_mode=VisionRunningMode.IMAGE, num_poses=1,
                min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, 
                min_tracking_confidence=0.5, output_segmentation_masks=False)
            pose_landmarker_obj = vision.PoseLandmarker.create_from_options(options_pose)
            print(f"Pose Landmarker loaded from buffer: {absolute_model_file_path}")
        except Exception as e:
            print(f"Error loading Pose Landmarker: {e}"); traceback.print_exc()
            return None, None
    return face_mesh, pose_landmarker_obj

def get_shoulder_roi_for_optical_flow(image_for_roi, pose_landmarker_obj):
    image_rgb = cv2.cvtColor(image_for_roi, cv2.COLOR_BGR2RGB)
    height, width = image_for_roi.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = pose_landmarker_obj.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        return None, None
    
    landmarks = detection_result.pose_landmarks[0]
    left_shoulder, right_shoulder = landmarks[11], landmarks[12]
    
    if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
        return None, None
    
    left_shoulder_x = int(left_shoulder.x * width)
    left_shoulder_y = int(left_shoulder.y * height)
    right_shoulder_x = int(right_shoulder.x * width)
    right_shoulder_y = int(right_shoulder.y * height)
    
    left_x1 = max(0, left_shoulder_x - config.SHOULDER_ROI_WIDTH//2)
    left_y1 = max(0, left_shoulder_y - config.SHOULDER_ROI_HEIGHT//2)
    left_x2 = min(width, left_shoulder_x + config.SHOULDER_ROI_WIDTH//2)
    left_y2 = min(height, left_shoulder_y + config.SHOULDER_ROI_HEIGHT//2)
    
    right_x1 = max(0, right_shoulder_x - config.SHOULDER_ROI_WIDTH//2)
    right_y1 = max(0, right_shoulder_y - config.SHOULDER_ROI_HEIGHT//2)
    right_x2 = min(width, right_shoulder_x + config.SHOULDER_ROI_WIDTH//2)
    right_y2 = min(height, right_shoulder_y + config.SHOULDER_ROI_HEIGHT//2)
    
    if ((left_x2 - left_x1) <= 10 or (left_y2 - left_y1) <= 10 or
        (right_x2 - right_x1) <= 10 or (right_y2 - right_y1) <= 10):
        return None, None
    
    left_shoulder_roi = (left_x1, left_y1, left_x2, left_y2)
    right_shoulder_roi = (right_x1, right_y1, right_x2, right_y2)
    
    return left_shoulder_roi, right_shoulder_roi

def get_forehead_roi(frame_display_orig_size, face_landmarks, actual_width, actual_height):
    face_pts = np.array([[int(lm.x*actual_width), int(lm.y*actual_height)] for lm in face_landmarks.landmark])
    fx_min,fy_min=np.min(face_pts,axis=0); fx_max,fy_max=np.max(face_pts,axis=0)
    fh_w,fh_h = fx_max-fx_min, fy_max-fy_min

    if fh_w > 0 and fh_h > 0:
        forehead_y_s = fy_min + int(fh_h * config.FOREHEAD_ROI_Y_START_FACTOR)
        forehead_y_e = fy_min + int(fh_h * config.FOREHEAD_ROI_Y_END_FACTOR)
        forehead_x_s = fx_min + int(fh_w * config.FOREHEAD_ROI_X_START_FACTOR)
        forehead_x_e = fx_min + int(fh_w * config.FOREHEAD_ROI_X_END_FACTOR)
        
        if forehead_y_s < forehead_y_e and forehead_x_s < forehead_x_e:
            return (forehead_x_s, forehead_y_s, forehead_x_e, forehead_y_e)
    return None
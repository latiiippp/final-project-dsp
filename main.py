import cv2
import numpy as np
import collections
import time
import os
import traceback

import config # Pastikan config diimpor
import utils
import mediapipe_utils
import rppg_processing
import respiration_processing
import display_utils
import analysis_utils

def initialize_camera_and_parameters(camera_id): # camera_id sekarang adalah parameter wajib
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (ID: {camera_id}).")
        return None, {}
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Request 1280x720
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30) # Request 30 FPS
    
    params = {}
    params['actual_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['actual_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['actual_fps'] = cap.get(cv2.CAP_PROP_FPS)

    if params['actual_width'] == 0: params['actual_width'] = config.DEFAULT_CAM_WIDTH
    if params['actual_height'] == 0: params['actual_height'] = config.DEFAULT_CAM_HEIGHT
    if params['actual_fps'] == 0 or params['actual_fps'] > 200: 
         params['actual_fps'] = config.DEFAULT_CAM_FPS
    
    params['fs'] = params['actual_fps']
    params['eps'] = 1e-9
    params['start_capture_time'] = time.time()
    
    print(f"Webcam source. Resolution: {params['actual_width']}x{params['actual_height']} at {int(params['actual_fps'])} fps, using Camera ID: {camera_id}")
    return cap, params

def initialize_buffers_and_state(fs):
    buffers = {}
    # rPPG Buffers
    buffer_size_rgb = int(fs * config.RPPG_RGB_BUFFER_SECONDS)
    buffers['r_signal_list'] = collections.deque(maxlen=buffer_size_rgb)
    buffers['g_signal_list'] = collections.deque(maxlen=buffer_size_rgb)
    buffers['b_signal_list'] = collections.deque(maxlen=buffer_size_rgb)
    buffers['collected_rppg_timestamps'] = [] 
    buffers['pos_display_buffer_size'] = int(fs * config.RPPG_POS_DISPLAY_SECONDS)
    buffers['rppg_pos_display_buffer'] = collections.deque(maxlen=buffers['pos_display_buffer_size'])
    buffers['current_hr'] = 0.0
    buffers['min_frames_for_pos_output'] = int(1.6*fs) + int(fs*1)

    # Respiration Buffers (Optical Flow)
    buffers['lk_params_of'] = dict(
        winSize=config.RESP_OF_LK_WIN_SIZE, 
        maxLevel=config.RESP_OF_LK_MAX_LEVEL, 
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, config.RESP_OF_LK_CRITERIA_COUNT, config.RESP_OF_LK_CRITERIA_EPS)
    )
    buffers['features_of'] = None
    buffers['old_gray_of'] = None
    buffers['roi_coords_of'] = None 
    buffers['valid_shoulders'] = (False, False)
    buffers['respiration_of_raw_signal_list'] = collections.deque(maxlen=int(fs*config.RESP_OF_RAW_BUFFER_SECONDS))
    buffers['collected_respiration_of_timestamps'] = []
    buffers['respiration_filtered_display_buffer'] = collections.deque(maxlen=int(fs*config.RESP_OF_FILTERED_DISPLAY_SECONDS))
    buffers['current_rpm'] = 0.0
    buffers['min_frames_for_resp_output'] = int(fs*5)
    return buffers

def cleanup_resources(cap, video_writer, pose_landmarker_obj, face_mesh):
    if pose_landmarker_obj and hasattr(pose_landmarker_obj, 'close'): pose_landmarker_obj.close()
    if face_mesh and hasattr(face_mesh, 'close'): face_mesh.close()
    if cap: cap.release()
    if video_writer: video_writer.release()
    cv2.destroyAllWindows()

def run_application(): # Hapus parameter camera_id dari sini
    # Gunakan camera_id dari config.py
    cap, cam_params = initialize_camera_and_parameters(config.CAMERA_ID) 
    if not cap: 
        return

    face_mesh, pose_landmarker_obj = mediapipe_utils.initialize_mediapipe_models(config.RPPG_MODEL_FILE_NAME)
    if not face_mesh or not pose_landmarker_obj:
        if cap: cap.release()
        return

    buffers_state = initialize_buffers_and_state(cam_params['fs'])
    display_config_live = display_utils.setup_display_and_video_writer(cam_params['actual_width'], cam_params['actual_height'], cam_params['fs'])
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("Cannot read frame from webcam. Exiting.")
                break
            frame_count += 1
            current_time_from_start = time.time() - cam_params['start_capture_time']

            frame_orig_for_pose_of = frame.copy() 
            frame_display_orig_size = cv2.flip(frame, 1)
            
            resized_frame_display = cv2.resize(frame_display_orig_size, 
                                               (display_config_live['display_camera_portion_w'], display_config_live['display_camera_h']))
            
            display_utils.clear_and_draw_graph_grids(display_config_live)

            current_hr, resized_frame_display_after_rppg = rppg_processing.process_rppg_frame(
                frame_display_orig_size, face_mesh, buffers_state, cam_params, 
                display_config_live, current_time_from_start, resized_frame_display
            )
            
            current_rpm, resized_frame_display_after_resp = respiration_processing.process_respiration_frame(
                frame_orig_for_pose_of, pose_landmarker_obj, buffers_state, cam_params, 
                display_config_live, current_time_from_start, frame_count, resized_frame_display_after_rppg
            )
            
            display_utils.assemble_and_show_display(
                config.DISPLAY_BACKGROUND_COLOR, 
                resized_frame_display_after_resp, 
                display_config_live, 
                current_hr, 
                current_rpm
            )

            if cv2.waitKey(1)&0xFF==ord('q') or cv2.getWindowProperty(display_config_live['window_name'],cv2.WND_PROP_VISIBLE)<1: 
                break
    
    except Exception as e:
        print(f"Error in main loop on frame {frame_count}: {e}")
        traceback.print_exc()
        buffers_state['current_hr'] = 0.0 
        buffers_state['current_rpm'] = 0.0
        buffers_state['features_of'] = None 
    finally:
        cleanup_resources(cap, display_config_live.get('video_writer'), pose_landmarker_obj, face_mesh)
        analysis_utils.generate_final_plots(buffers_state, cam_params, display_config_live)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created '{models_dir}' directory. Please place '{config.RPPG_MODEL_FILE_NAME}' or similar (e.g. 'pose_landmarker_lite.task') there.")
    
    run_application() # Panggil tanpa argumen, karena camera_id diambil dari config
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import collections
from scipy import signal
import matplotlib.pyplot as plt
import time
import os
import traceback

# --- Constants and Configuration ---
# (These were previously embedded or calculated early in main)
# Note: Some 'constants' like FPS depend on camera initialization
RPPG_MODEL_FILE_NAME = "pose_landmarker.task"
STANDARD_SIZE_OF_FOR_OPTICAL_FLOW = (640, 480)

# --- Utility and Algorithm Functions ---
def draw_grid(image, graph_width, graph_height, color=(200, 200, 200), thickness=1):
    for i in range(1, 5): 
        y = int(i * graph_height / 5)
        cv2.line(image, (0, y), (graph_width, y), color, thickness)
    for i in range(1, 10): 
        x = int(i * graph_width / 10)
        cv2.line(image, (x, 0), (x, graph_height), color, thickness)

def cpu_POS(rgb_signal_segment, **kargs):
    eps = 10**-9
    X = rgb_signal_segment 
    if X.ndim == 2 and X.shape[0] == 3: X = np.expand_dims(X, axis=0)
    elif X.ndim != 3 or X.shape[0] != 1 or X.shape[1] != 3:
        raise ValueError(f"Input signal for cpu_POS must be shape (1,3,N) or (3,N), got {X.shape}")
    e, c, f = X.shape
    fps = kargs.get('fps', 30.0) 
    if fps <= 0: fps = 30.0
    w = int(1.6 * fps)   
    if w == 0: w = 1 
    if f < w : return np.zeros((e, f))
    P = np.array([[0,1,-1],[-2,1,1]]); Q = np.stack([P for _ in range(e)],axis=0)
    H = np.zeros((e,f)) 
    for n in np.arange(w-1,f): 
        m=n-w+1; Cn=X[:,:,m:(n+1)]; M=1.0/(np.mean(Cn,axis=2)+eps)
        Cn_norm=np.multiply(np.expand_dims(M,axis=2),Cn)
        S=np.diagonal(np.tensordot(Q,Cn_norm,axes=([2],[1])),axis1=0,axis2=2).T
        S1=S[:,0,:]; S2=S[:,1,:]; alpha=np.std(S1,axis=1)/(eps+np.std(S2,axis=1))
        Hn_w=S1+np.expand_dims(alpha,axis=1)*S2
        Hnm_w=Hn_w-np.expand_dims(np.mean(Hn_w,axis=1),axis=1)
        if Hnm_w.shape[1]>0: H[:,n]=Hnm_w[:,-1] 
    return H

def get_roi_for_optical_flow_from_snippet_logic(image_for_roi, pose_landmarker_obj, x_size=100, y_size=100, shift_x=0, shift_y=0):
    image_rgb = cv2.cvtColor(image_for_roi, cv2.COLOR_BGR2RGB)
    height, width = image_for_roi.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = pose_landmarker_obj.detect(mp_image)
    if not detection_result.pose_landmarks: return None
    landmarks = detection_result.pose_landmarks[0] 
    left_shoulder, right_shoulder = landmarks[11], landmarks[12]
    if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5: return None
    center_x = int((left_shoulder.x + right_shoulder.x) * width / 2) + shift_x
    center_y = int((left_shoulder.y + right_shoulder.y) * height / 2) + shift_y
    left_x_roi, right_x_roi = max(0, center_x - x_size), min(width, center_x + x_size)
    top_y_roi, bottom_y_roi = max(0, center_y - y_size), min(height, center_y)
    if (right_x_roi - left_x_roi) <= 10 or (bottom_y_roi - top_y_roi) <= 10: return None
    return (left_x_roi, top_y_roi, right_x_roi, bottom_y_roi)

# Fungsi baru untuk deteksi ROI bahu secara spesifik untuk optical flow
def get_shoulder_roi_for_optical_flow(image_for_roi, pose_landmarker_obj, shoulder_width=60, shoulder_height=40):
    """
    Mendeteksi ROI bahu kiri dan kanan untuk algoritma Lucas-Kanade Optical Flow.
    
    Args:
        image_for_roi: Frame gambar untuk deteksi
        pose_landmarker_obj: MediaPipe Pose Landmarker object
        shoulder_width: Lebar area ROI untuk setiap bahu (default: 60)
        shoulder_height: Tinggi area ROI untuk setiap bahu (default: 40)
    
    Returns:
        Tuple of (left_shoulder_roi, right_shoulder_roi) dimana masing-masing adalah (x1, y1, x2, y2)
        atau (None, None) jika deteksi gagal
    """
    image_rgb = cv2.cvtColor(image_for_roi, cv2.COLOR_BGR2RGB)
    height, width = image_for_roi.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = pose_landmarker_obj.detect(mp_image)
    
    if not detection_result.pose_landmarks:
        return None, None
    
    landmarks = detection_result.pose_landmarks[0]
    # Landmark 11: Left Shoulder, Landmark 12: Right Shoulder
    left_shoulder, right_shoulder = landmarks[11], landmarks[12]
    
    # Periksa visibilitas
    if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
        return None, None
    
    # Konversi koordinat landmark ke pixel
    left_shoulder_x = int(left_shoulder.x * width)
    left_shoulder_y = int(left_shoulder.y * height)
    right_shoulder_x = int(right_shoulder.x * width)
    right_shoulder_y = int(right_shoulder.y * height)
    
    # Tentukan ROI untuk bahu kiri
    left_x1 = max(0, left_shoulder_x - shoulder_width//2)
    left_y1 = max(0, left_shoulder_y - shoulder_height//2)
    left_x2 = min(width, left_shoulder_x + shoulder_width//2)
    left_y2 = min(height, left_shoulder_y + shoulder_height//2)
    
    # Tentukan ROI untuk bahu kanan
    right_x1 = max(0, right_shoulder_x - shoulder_width//2)
    right_y1 = max(0, right_shoulder_y - shoulder_height//2)
    right_x2 = min(width, right_shoulder_x + shoulder_width//2)
    right_y2 = min(height, right_shoulder_y + shoulder_height//2)
    
    # Validasi ukuran ROI
    if ((left_x2 - left_x1) <= 10 or (left_y2 - left_y1) <= 10 or
        (right_x2 - right_x1) <= 10 or (right_y2 - right_y1) <= 10):
        return None, None
    
    left_shoulder_roi = (left_x1, left_y1, left_x2, left_y2)
    right_shoulder_roi = (right_x1, right_y1, right_x2, right_y2)
    
    return left_shoulder_roi, right_shoulder_roi

def initialize_optical_flow_features_from_snippet_logic(frame_for_of, pose_landmarker_obj, lk_params_dict):
    roi_coords = get_roi_for_optical_flow_from_snippet_logic(frame_for_of, pose_landmarker_obj)
    if roi_coords is None: return None, None, None
    l_x, t_y, r_x, b_y = roi_coords
    gray_frame_for_of = cv2.cvtColor(frame_for_of, cv2.COLOR_BGR2GRAY)
    roi_gray = gray_frame_for_of[t_y:b_y, l_x:r_x]
    if roi_gray.size == 0 or roi_gray.shape[0] < 5 or roi_gray.shape[1] < 5: return None, None, None
    features_to_track = cv2.goodFeaturesToTrack(roi_gray, maxCorners=50, qualityLevel=0.2, minDistance=5, blockSize=3)
    if features_to_track is None: return None, None, None
    features_to_track = np.float32(features_to_track)
    features_to_track[:,:,0] += l_x 
    features_to_track[:,:,1] += t_y 
    return features_to_track, gray_frame_for_of.copy(), roi_coords

def initialize_shoulder_optical_flow_features(frame_for_of, pose_landmarker_obj, lk_params_dict):
    """
    Menginisialisasi fitur optical flow pada area bahu untuk algoritma Lucas-Kanade.
    
    Args:
        frame_for_of: Frame gambar untuk inisialisasi
        pose_landmarker_obj: MediaPipe Pose Landmarker object
        lk_params_dict: Parameter untuk algoritma Lucas-Kanade
    
    Returns:
        Tuple (features_to_track, gray_frame, shoulder_rois, valid_shoulders)
        - features_to_track: Array fitur untuk ditelusuri
        - gray_frame: Frame grayscale untuk optical flow
        - shoulder_rois: (roi_left, roi_right) koordinat ROI bahu kiri dan kanan
        - valid_shoulders: (is_left_valid, is_right_valid) status validitas bahu
    """
    left_roi, right_roi = get_shoulder_roi_for_optical_flow(frame_for_of, pose_landmarker_obj)
    if left_roi is None and right_roi is None:
        return None, None, (None, None), (False, False)
    
    gray_frame_for_of = cv2.cvtColor(frame_for_of, cv2.COLOR_BGR2GRAY)
    combined_features = []
    valid_shoulders = [False, False]  # [left_valid, right_valid]
    
    # Proses bahu kiri jika terdeteksi
    if left_roi is not None:
        l_x, l_y, r_x, r_y = left_roi
        left_roi_gray = gray_frame_for_of[l_y:r_y, l_x:r_x]
        if left_roi_gray.size > 0 and left_roi_gray.shape[0] >= 5 and left_roi_gray.shape[1] >= 5:
            left_features = cv2.goodFeaturesToTrack(
                left_roi_gray, maxCorners=25, qualityLevel=0.2, minDistance=3, blockSize=3
            )
            if left_features is not None:
                left_features = np.float32(left_features)
                left_features[:,:,0] += l_x
                left_features[:,:,1] += l_y
                combined_features.append(left_features)
                valid_shoulders[0] = True
    
    # Proses bahu kanan jika terdeteksi
    if right_roi is not None:
        l_x, l_y, r_x, r_y = right_roi
        right_roi_gray = gray_frame_for_of[l_y:r_y, l_x:r_x]
        if right_roi_gray.size > 0 and right_roi_gray.shape[0] >= 5 and right_roi_gray.shape[1] >= 5:
            right_features = cv2.goodFeaturesToTrack(
                right_roi_gray, maxCorners=25, qualityLevel=0.2, minDistance=3, blockSize=3
            )
            if right_features is not None:
                right_features = np.float32(right_features)
                right_features[:,:,0] += l_x
                right_features[:,:,1] += l_y
                combined_features.append(right_features)
                valid_shoulders[1] = True
    
    # Jika tidak ada bahu yang valid, return None
    if not combined_features:
        return None, None, (left_roi, right_roi), (False, False)
    
    # Gabungkan semua fitur yang terdeteksi
    all_features = np.vstack(combined_features)
    
    return all_features, gray_frame_for_of.copy(), (left_roi, right_roi), tuple(valid_shoulders)

# --- Initialization Functions ---
def initialize_camera_and_parameters(camera_id=2):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (ID: {camera_id}).")
        return None, {}
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    params = {}
    params['actual_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['actual_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['actual_fps'] = cap.get(cv2.CAP_PROP_FPS)

    if params['actual_width'] == 0: params['actual_width'] = 640
    if params['actual_height'] == 0: params['actual_height'] = 480
    if params['actual_fps'] == 0 or params['actual_fps'] > 200: params['actual_fps'] = 30
    
    params['fs'] = params['actual_fps']
    params['eps'] = 1e-9
    params['start_capture_time'] = time.time()
    
    print(f"Webcam source. Resolution: {params['actual_width']}x{params['actual_height']} at {int(params['actual_fps'])} fps")
    return cap, params

def initialize_mediapipe(model_file_name):
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_model_file_path = os.path.join(script_dir, "models", model_file_name)
    
    pose_landmarker_obj = None
    if not os.path.exists(absolute_model_file_path):
        print(f"Error: Pose landmarker model not found at {absolute_model_file_path}")
        # Instructions to user...
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

def initialize_buffers_and_state(fs):
    buffers = {}
    # rPPG
    buffer_size_rgb = int(fs * 15)
    buffers['r_signal_list'] = collections.deque(maxlen=buffer_size_rgb)
    buffers['g_signal_list'] = collections.deque(maxlen=buffer_size_rgb)
    buffers['b_signal_list'] = collections.deque(maxlen=buffer_size_rgb)
    buffers['collected_rppg_timestamps'] = [] # Changed to list
    buffers['pos_display_buffer_size'] = int(fs * 10)
    buffers['rppg_pos_display_buffer'] = collections.deque(maxlen=buffers['pos_display_buffer_size'])
    buffers['pos_lowcut_hr']=0.75; buffers['pos_highcut_hr']=3.0; buffers['pos_filter_order']=3
    buffers['min_frames_for_pos_output'] = int(1.6*fs) + int(fs*1)
    buffers['current_hr'] = 0.0    # Respiration - menggunakan algoritma Lucas-Kanade untuk tracking pergerakan bahu
    # Parameter untuk algoritma Lucas-Kanade
    buffers['lk_params_of'] = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Variabel untuk tracking bahu
    buffers['features_of'] = None  # Fitur yang dilacak
    buffers['old_gray_of'] = None  # Frame grayscale sebelumnya
    buffers['roi_coords_of'] = None  # (left_roi, right_roi) koordinat ROI bahu kiri dan kanan
    buffers['valid_shoulders'] = (False, False)  # Status validitas bahu (left_valid, right_valid)
    
    # Buffer untuk sinyal respirasi
    buffers['respiration_of_raw_signal_list'] = collections.deque(maxlen=int(fs*20))
    buffers['collected_respiration_of_timestamps'] = []  # Menyimpan timestamp
    buffers['respiration_filtered_display_buffer'] = collections.deque(maxlen=buffers['pos_display_buffer_size'])
    
    # Parameter filter dan deteksi respirasi
    # Frekuensi cutoff: 0.1-0.5 Hz sesuai dengan 6-30 respirasi/menit
    buffers['resp_lowcut_hz']=0.1
    buffers['resp_highcut_hz']=0.5
    buffers['resp_filter_order']=2
    buffers['min_frames_for_resp_output'] = int(fs*5)
    buffers['current_rpm'] = 0.0
    return buffers

def setup_display_and_video(actual_width, actual_height, fs):
    display_config = {}
    display_config['graph_height'] = 120
    display_config['panel_padding'] = 10
    display_config['graph_title_space'] = 20
    display_config['graph_text_color'] = (0,0,0)
    display_config['grid_color'] = (220,220,220)

    # Original layout logic from user's file
    display_config['TOTAL_WINDOW_WIDTH'] = actual_width + (380 + 2 * display_config['panel_padding'])
    if display_config['TOTAL_WINDOW_WIDTH'] < 800: display_config['TOTAL_WINDOW_WIDTH'] = 800
    
    display_config['display_camera_portion_w'] = int(display_config['TOTAL_WINDOW_WIDTH'] * 0.35)
    display_config['display_graphs_portion_w'] = display_config['TOTAL_WINDOW_WIDTH'] - display_config['display_camera_portion_w']
    
    camera_aspect_ratio = actual_height / actual_width
    display_config['display_camera_h'] = int(display_config['display_camera_portion_w'] * camera_aspect_ratio)
    
    display_config['graph_width'] = display_config['display_graphs_portion_w'] - 2 * display_config['panel_padding']
    if display_config['graph_width'] < 100: display_config['graph_width'] = 100

    graphs_area_h = (display_config['graph_height'] + display_config['graph_title_space']) * 4 + \
                    display_config['panel_padding'] * 3 + 40
    display_config['graphs_area_height_calc'] = graphs_area_h # Store for later use in y-positioning
    display_config['combined_height'] = max(display_config['display_camera_h'], graphs_area_h)
    display_config['combined_width'] = display_config['TOTAL_WINDOW_WIDTH']
    
    display_config['window_name'] = 'rPPG (POS) & Respiration (Optical Flow) - 35/65 Layout'
    cv2.namedWindow(display_config['window_name'])
    
    gh, gw = display_config['graph_height'], display_config['graph_width']
    display_config['rppg_raw_g'] = np.ones((gh,gw,3),dtype=np.uint8)*255
    display_config['rppg_filt_g'] = np.ones((gh,gw,3),dtype=np.uint8)*255
    display_config['resp_raw_g'] = np.ones((gh,gw,3),dtype=np.uint8)*255
    display_config['resp_filt_g'] = np.ones((gh,gw,3),dtype=np.uint8)*255
    
    output_video_filename = "output_rppg_respiration_resized_layout.mp4"
    display_config['video_writer'] = cv2.VideoWriter(output_video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 
                                                     float(fs), (display_config['combined_width'], display_config['combined_height']))
    display_config['r_panel_x'] = display_config['display_camera_portion_w']
    return display_config

# --- Per-Frame Processing Functions ---
def process_rppg_frame(frame_display_orig_size, face_mesh, buffers, cam_params, display_config, current_time_from_start, resized_frame_display_in):
    actual_width, actual_height = cam_params['actual_width'], cam_params['actual_height']
    fs, eps = cam_params['fs'], cam_params['eps']
    resized_frame_display = resized_frame_display_in.copy() # Work on a copy

    rgb_frame_mp_face = cv2.cvtColor(frame_display_orig_size, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame_mp_face)
    valid_roi_for_rppg = False

    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        face_pts = np.array([[int(lm.x*actual_width), int(lm.y*actual_height)] for lm in face_landmarks.landmark])
        fx_min,fy_min=np.min(face_pts,axis=0); fx_max,fy_max=np.max(face_pts,axis=0)
        fh_w,fh_h = fx_max-fx_min, fy_max-fy_min
        if fh_w > 0 and fh_h > 0:
            forehead_y_s, forehead_y_e = fy_min+int(fh_h*0.1), fy_min+int(fh_h*0.25)
            forehead_x_s, forehead_x_e = fx_min+int(fh_w*0.2), fx_min+int(fh_w*0.8)
            forehead_roi_disp_orig = frame_display_orig_size[forehead_y_s:forehead_y_e, forehead_x_s:forehead_x_e]
            if forehead_roi_disp_orig.size > 100: 
                b,g,r = np.mean(forehead_roi_disp_orig[:,:,0]), np.mean(forehead_roi_disp_orig[:,:,1]), np.mean(forehead_roi_disp_orig[:,:,2])
                buffers['r_signal_list'].append(r); buffers['g_signal_list'].append(g); buffers['b_signal_list'].append(b)
                buffers['collected_rppg_timestamps'].append(current_time_from_start)
                valid_roi_for_rppg = True
                scale_x_roi_disp = display_config['display_camera_portion_w'] / actual_width
                scale_y_roi_disp = display_config['display_camera_h'] / actual_height
                rx_s, ry_s = int(forehead_x_s * scale_x_roi_disp), int(forehead_y_s * scale_y_roi_disp)
                rx_e, ry_e = int(forehead_x_e * scale_x_roi_disp), int(forehead_y_e * scale_y_roi_disp)
                cv2.rectangle(resized_frame_display, (rx_s, ry_s), (rx_e, ry_e), (255,0,0),1)
    
    # Heart rate calculation and graph drawing
    buffers['current_hr'] = 0.0 # Reset for this frame
    if valid_roi_for_rppg and len(buffers['g_signal_list']) >= buffers['min_frames_for_pos_output']:
        seg_len = min(len(buffers['g_signal_list']), buffers['pos_display_buffer_size'])
        r_s,g_s,b_s = list(buffers['r_signal_list'])[-seg_len:], list(buffers['g_signal_list'])[-seg_len:], list(buffers['b_signal_list'])[-seg_len:]
        if len(r_s) > int(1.6*fs): 
            pos_out_full_seg = cpu_POS(np.array([r_s,g_s,b_s]).reshape(1,3,-1), fps=fs)[0]
            buffers['rppg_pos_display_buffer'].clear(); buffers['rppg_pos_display_buffer'].extend(pos_out_full_seg)
            if len(buffers['rppg_pos_display_buffer']) > int(fs*2): 
                det_pos = signal.detrend(list(buffers['rppg_pos_display_buffer']))
                nyq,low,high = 0.5*fs, buffers['pos_lowcut_hr']/(0.5*fs), buffers['pos_highcut_hr']/(0.5*fs)
                if low>0 and high <1 and low < high:
                    bf,af = signal.butter(buffers['pos_filter_order'], [low,high], btype='band')
                    filt_pos = signal.filtfilt(bf,af,det_pos)
                    peaks,_ = signal.find_peaks((filt_pos-np.mean(filt_pos))/(np.std(filt_pos)+eps), prominence=0.4, distance=fs/(buffers['pos_highcut_hr']*2.0)) 
                    if len(peaks)>=2: buffers['current_hr'] = np.clip(60.0/(np.mean(np.diff(peaks))/fs), 35, 180)
                    min_v,max_v=np.min(filt_pos),np.max(filt_pos)
                    if max_v-min_v > eps:
                        norm_disp=(filt_pos-min_v)/(max_v-min_v); pts=min(display_config['graph_width'],len(norm_disp))
                        for i in range(1,pts): cv2.line(display_config['rppg_filt_g'],(i-1,display_config['graph_height']-int(norm_disp[i-1]*display_config['graph_height'])),(i,display_config['graph_height']-int(norm_disp[i]*display_config['graph_height'])),(0,165,255),1) 
        if len(buffers['rppg_pos_display_buffer'])>1:
            raw_plot_data = np.array(list(buffers['rppg_pos_display_buffer']))
            min_v,max_v=np.min(raw_plot_data),np.max(raw_plot_data)
            if max_v-min_v > eps:
                norm_disp=(raw_plot_data-min_v)/(max_v-min_v); pts=min(display_config['graph_width'],len(norm_disp))
                for i in range(1,pts): cv2.line(display_config['rppg_raw_g'],(i-1,display_config['graph_height']-int(norm_disp[i-1]*display_config['graph_height'])),(i,display_config['graph_height']-int(norm_disp[i]*display_config['graph_height'])),(255,0,0),1) 
    elif not valid_roi_for_rppg: cv2.putText(display_config['rppg_raw_g'],"No Face ROI",(10,display_config['graph_height']//2),cv2.FONT_HERSHEY_SIMPLEX,0.4,display_config['graph_text_color'],1)
    else: cv2.putText(display_config['rppg_raw_g'],"Buffering RGB...",(10,display_config['graph_height']//2),cv2.FONT_HERSHEY_SIMPLEX,0.4,display_config['graph_text_color'],1)
    
    return buffers['current_hr'], resized_frame_display

def process_respiration_frame(frame_orig_for_pose_of, pose_landmarker_obj, buffers, cam_params, display_config, current_time_from_start, frame_count, resized_frame_display_in):
    fs, eps = cam_params['fs'], cam_params['eps']
    resized_frame_display = resized_frame_display_in.copy() # Work on a copy

    frame_for_of_processing = cv2.resize(frame_orig_for_pose_of, STANDARD_SIZE_OF_FOR_OPTICAL_FLOW)
    
    # Manage features_of dan old_gray_of untuk deteksi pergerakan bahu
    if buffers['features_of'] is None or len(buffers['features_of']) < 10:
        if frame_count > 1:
            cv2.putText(display_config['resp_raw_g'], "Re-Init Shoulders", (10, display_config['graph_height']//2-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)
        
        # Gunakan fungsi baru untuk melacak bahu
        features, gray_frame, shoulder_rois, valid_shoulders = initialize_shoulder_optical_flow_features(
            frame_for_of_processing, pose_landmarker_obj, buffers['lk_params_of']
        )
        
        buffers['features_of'] = features
        buffers['old_gray_of'] = gray_frame
        buffers['roi_coords_of'] = shoulder_rois  # Sekarang berisi (left_roi, right_roi)
        buffers['valid_shoulders'] = valid_shoulders  # Status validitas bahu (left_valid, right_valid)
        
        if buffers['features_of'] is None:
            cv2.putText(display_config['resp_raw_g'], "Shoulder Track Failed", (10, display_config['graph_height']//2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)
        else:
            cv2.putText(display_config['resp_raw_g'], "Shoulder Track OK", (10, display_config['graph_height']//2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)

    if buffers['features_of'] is not None and buffers['old_gray_of'] is not None:
        frame_gray_of_processing = cv2.cvtColor(frame_for_of_processing, cv2.COLOR_BGR2GRAY)
        new_features, status, error = cv2.calcOpticalFlowPyrLK(buffers['old_gray_of'], frame_gray_of_processing, 
                                                              buffers['features_of'], None, **buffers['lk_params_of'])
        good_new = []
        if new_features is not None and status is not None:
            good_new = new_features[status.flatten() == 1]
        
        # Gambar ROI dan fitur bahu
        left_roi, right_roi = buffers['roi_coords_of'] if isinstance(buffers['roi_coords_of'], tuple) else (None, None)
        
        # Faktor skala untuk menampilkan ROI pada frame yang ditampilkan
        scale_x_of_to_disp = display_config['display_camera_portion_w'] / STANDARD_SIZE_OF_FOR_OPTICAL_FLOW[0]
        scale_y_of_to_disp = display_config['display_camera_h'] / STANDARD_SIZE_OF_FOR_OPTICAL_FLOW[1]
        
        # Gambar ROI bahu kiri (jika valid)
        if left_roi:
            l_x, l_y, r_x, r_y = left_roi
            l_d = int(l_x * scale_x_of_to_disp)
            t_d = int(l_y * scale_y_of_to_disp)
            r_d = int(r_x * scale_x_of_to_disp)
            b_d = int(r_y * scale_y_of_to_disp)
            cv2.rectangle(resized_frame_display, 
                         (display_config['display_camera_portion_w'] - r_d, t_d), 
                         (display_config['display_camera_portion_w'] - l_d, b_d), 
                         (0, 255, 0), 1)
            cv2.putText(resized_frame_display, "L", 
                       (display_config['display_camera_portion_w'] - r_d + 5, t_d + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Gambar ROI bahu kanan (jika valid)
        if right_roi:
            l_x, l_y, r_x, r_y = right_roi
            l_d = int(l_x * scale_x_of_to_disp)
            t_d = int(l_y * scale_y_of_to_disp)
            r_d = int(r_x * scale_x_of_to_disp)
            b_d = int(r_y * scale_y_of_to_disp)
            cv2.rectangle(resized_frame_display, 
                         (display_config['display_camera_portion_w'] - r_d, t_d), 
                         (display_config['display_camera_portion_w'] - l_d, b_d), 
                         (0, 255, 255), 1)
            cv2.putText(resized_frame_display, "R", 
                       (display_config['display_camera_portion_w'] - r_d + 5, t_d + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Gambar fitur yang dilacak
        for pt_new_abs in good_new:
            x_abs_std, y_abs_std = pt_new_abs.ravel()
            disp_x_on_resized = int(x_abs_std * scale_x_of_to_disp)
            disp_y_on_resized = int(y_abs_std * scale_y_of_to_disp)
            cv2.circle(resized_frame_display, 
                      (display_config['display_camera_portion_w'] - disp_x_on_resized, disp_y_on_resized), 
                      2, (0, 255, 0), -1)
        
        if len(good_new) > 5:
            # Hitung rata-rata posisi y dari fitur sebagai sinyal respirasi
            avg_y_of = np.mean(good_new[:, 0, 1])
            buffers['respiration_of_raw_signal_list'].append(avg_y_of)
            buffers['collected_respiration_of_timestamps'].append(current_time_from_start)
            buffers['features_of'] = good_new.reshape(-1, 1, 2)
            buffers['old_gray_of'] = frame_gray_of_processing.copy()
        else:
            buffers['features_of'] = None
            if buffers['old_gray_of'] is not None:
                cv2.putText(display_config['resp_raw_g'], "OF Track Lost", (10, display_config['graph_height']//2+10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)
      # RPM calculation and graph drawing
    buffers['current_rpm'] = 0.0 # Reset for this frame
    if len(buffers['respiration_of_raw_signal_list']) >= buffers['min_frames_for_resp_output']:
        resp_seg_of = np.array(list(buffers['respiration_of_raw_signal_list'])[-buffers['pos_display_buffer_size']:]) # Use consistent display length
        if len(resp_seg_of) > int(fs*2): 
            # Detren sinyal untuk menghilangkan drift jangka panjang
            det_resp_of = signal.detrend(resp_seg_of)
            
            # Filter sinyal untuk memperoleh sinyal pernafasan
            # Frekuensi pernafasan normal: 0.1 - 0.5 Hz (6-30 respirasi per menit)
            nyq_r,low_r,high_r = 0.5*fs, buffers['resp_lowcut_hz']/(0.5*fs), buffers['resp_highcut_hz']/(0.5*fs)
            if low_r>0 and high_r<1 and low_r<high_r:
                # Butterworth bandpass filter
                br,ar = signal.butter(buffers['resp_filter_order'], [low_r,high_r], btype='band')
                filt_resp_of = signal.filtfilt(br,ar,det_resp_of)
                
                # Menyimpan sinyal yang difilter untuk ditampilkan
                buffers['respiration_filtered_display_buffer'].clear()
                buffers['respiration_filtered_display_buffer'].extend(filt_resp_of)
                
                # Mendeteksi puncak untuk menghitung laju pernapasan
                # Normalisasi sinyal untuk deteksi puncak yang lebih konsisten
                norm_sig = (filt_resp_of-np.mean(filt_resp_of))/(np.std(filt_resp_of)+eps)
                
                # Parameter prominence mengontrol seberapa menonjol suatu puncak
                # Distance memastikan jarak minimum antar puncak sesuai dengan frekuensi pernapasan maksimum
                peaks_r,_ = signal.find_peaks(norm_sig, prominence=0.35, distance=fs/(buffers['resp_highcut_hz']*2.5)) 
                
                # Hitung respirasi per menit (RPM) dari interval antar puncak
                if len(peaks_r)>=2: 
                    mean_period = np.mean(np.diff(peaks_r))/fs  # periode rata-rata dalam detik
                    buffers['current_rpm'] = np.clip(60.0/mean_period, 4, 30)  # konversi ke RPM dan batasi nilai
                
                # Visualisasi sinyal yang difilter
                if len(buffers['respiration_filtered_display_buffer'])>1:
                    data_plot = np.array(list(buffers['respiration_filtered_display_buffer']))
                    min_v,max_v=np.min(data_plot),np.max(data_plot)
                    if max_v-min_v > eps:
                        norm_disp=(data_plot-min_v)/(max_v-min_v)
                        pts=min(display_config['graph_width'],len(norm_disp))
                        # Gambar sinyal yang difilter dengan warna hijau
                        for i in range(1,pts): 
                            cv2.line(display_config['resp_filt_g'],
                                   (i-1,display_config['graph_height']-int(norm_disp[i-1]*display_config['graph_height'])),
                                   (i,display_config['graph_height']-int(norm_disp[i]*display_config['graph_height'])),
                                   (0,100,0),1)        # Plot sinyal bahu mentah untuk tampilan
        if len(buffers['respiration_of_raw_signal_list'])>1:
            data_plot_raw = np.array(list(buffers['respiration_of_raw_signal_list'])[-buffers['pos_display_buffer_size']:])
            min_v,max_v=np.min(data_plot_raw),np.max(data_plot_raw)
            if max_v-min_v > eps:
                norm_disp=(data_plot_raw-min_v)/(max_v-min_v)
                pts=min(display_config['graph_width'],len(norm_disp))
                # Gambar sinyal mentah dengan warna biru
                for i in range(1,pts): 
                    cv2.line(display_config['resp_raw_g'],
                           (i-1,display_config['graph_height']-int(norm_disp[i-1]*display_config['graph_height'])),
                           (i,display_config['graph_height']-int(norm_disp[i]*display_config['graph_height'])),
                           (0,0,255),1)
                
                # Tambahkan teks informasi RPM pada grafik
                if buffers['current_rpm'] > 0:
                    rpm_text = f"RPM: {buffers['current_rpm']:.1f}"
                    cv2.putText(display_config['resp_filt_g'], rpm_text,
                              (display_config['graph_width']-100, 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,0), 1)
    
    # Tampilkan status jika fitur pelacakan belum siap
    elif buffers['features_of'] is not None:
        cv2.putText(display_config['resp_raw_g'], "Collecting Shoulder Movement...",
                   (10, display_config['graph_height']//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)
    elif buffers['features_of'] is None and buffers['old_gray_of'] is not None:
        pass
    else:
        cv2.putText(display_config['resp_raw_g'], "Waiting Shoulder Detection...",
                   (10, display_config['graph_height']//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)
    
    return buffers['current_rpm'], resized_frame_display

def assemble_and_show_display(combined_display_bg, resized_frame_display, display_config, current_hr, current_rpm):
    # Place camera frame
    cam_y_offset = (display_config['combined_height'] - display_config['display_camera_h']) // 2
    if display_config['combined_height'] <= display_config['display_camera_h']: cam_y_offset = 0 # Ensure non-negative
    
    # Ensure camera frame fits
    cam_h_to_place = min(display_config['display_camera_h'], display_config['combined_height'] - cam_y_offset)
    cam_w_to_place = min(display_config['display_camera_portion_w'], display_config['combined_width'])
    
    if cam_h_to_place > 0 and cam_w_to_place > 0:
        # Resize again if necessary to fit the calculated safe dimensions
        if resized_frame_display.shape[0] != cam_h_to_place or resized_frame_display.shape[1] != cam_w_to_place:
            resized_frame_display_safe = cv2.resize(resized_frame_display, (cam_w_to_place, cam_h_to_place))
        else:
            resized_frame_display_safe = resized_frame_display
        combined_display_bg[cam_y_offset : cam_y_offset + cam_h_to_place, 0 : cam_w_to_place] = resized_frame_display_safe


    # Place graphs
    current_graph_y = (display_config['combined_height'] - display_config['graphs_area_height_calc']) // 2 + display_config['panel_padding']
    if display_config['combined_height'] <= display_config['graphs_area_height_calc']: current_graph_y = display_config['panel_padding']


    graphs_data = [
        (display_config['rppg_raw_g'], "Raw POS rPPG", None, 0),
        (display_config['rppg_filt_g'], "Filtered POS (HR)", f"HR: {current_hr:.1f} BPM" if current_hr > 0 else "HR: N/A", current_hr),
        (display_config['resp_raw_g'], "Raw Respiration (OF Y-avg)", None, 0),
        (display_config['resp_filt_g'], "Filtered Respiration (OF)", f"RPM: {current_rpm:.1f}" if current_rpm > 0 else "RPM: N/A", current_rpm)
    ]

    for img_to_place, title_str, val_text_str, val_num in graphs_data:
        if current_graph_y + display_config['graph_height'] <= display_config['combined_height']: # Check vertical fit
            y_start = current_graph_y
            y_end = y_start + display_config['graph_height']
            x_start = display_config['r_panel_x'] + display_config['panel_padding']
            x_end = x_start + display_config['graph_width']

            # Ensure graph image fits into the designated slice
            if y_end <= combined_display_bg.shape[0] and x_end <= combined_display_bg.shape[1] and \
               img_to_place.shape[0] == (y_end - y_start) and img_to_place.shape[1] == (x_end - x_start):
                combined_display_bg[y_start:y_end, x_start:x_end] = img_to_place
                cv2.putText(combined_display_bg,title_str,(x_start, y_start - 5),cv2.FONT_HERSHEY_SIMPLEX,0.4,display_config['graph_text_color'],1)
                if val_text_str: cv2.putText(combined_display_bg,val_text_str,(x_start + 5, y_end + 12),cv2.FONT_HERSHEY_SIMPLEX,0.45,display_config['graph_text_color'] if val_num > 0 else (100,100,100),1)
            current_graph_y += display_config['graph_height'] + display_config['graph_title_space'] + (15 if val_text_str else 0)
            
    cv2.imshow(display_config['window_name'], combined_display_bg)
    if display_config['video_writer'] is not None: display_config['video_writer'].write(combined_display_bg)


# --- Cleanup and Final Plotting ---
def cleanup_resources(cap, video_writer, pose_landmarker_obj, face_mesh):
    if pose_landmarker_obj and hasattr(pose_landmarker_obj, 'close'): pose_landmarker_obj.close()
    if face_mesh and hasattr(face_mesh, 'close'): face_mesh.close()
    if cap: cap.release()
    if video_writer: video_writer.release()
    cv2.destroyAllWindows()

def generate_final_plots(buffers, cam_params, display_config):
    fs = cam_params['fs']
    # rPPG Plot
    if len(buffers['g_signal_list']) > buffers['min_frames_for_pos_output']:
        print("Plotting final rPPG signals...")
        final_r,final_g,final_b = np.array(list(buffers['r_signal_list'])),np.array(list(buffers['g_signal_list'])),np.array(list(buffers['b_signal_list']))
        final_pos_full = cpu_POS(np.array([final_r,final_g,final_b]).reshape(1,3,-1),fps=fs)[0]
        times_rppg_full = np.array(buffers['collected_rppg_timestamps'][:len(final_pos_full)])
        
        plt.figure(figsize=(15,8)); plt.suptitle("rPPG (POS) Full Session Analysis", fontsize=14)
        plt.subplot(211); plt.plot(times_rppg_full,final_pos_full, label="Raw POS"); plt.title("Raw POS Signal"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend(); plt.grid(True)
        
        det_pos_full = signal.detrend(final_pos_full)
        nyq,low,high = 0.5*fs, buffers['pos_lowcut_hr']/(0.5*fs), buffers['pos_highcut_hr']/(0.5*fs)
        if low>0 and high <1 and low < high:
            bf,af = signal.butter(buffers['pos_filter_order'], [low,high], btype='band')
            filt_pos_full = signal.filtfilt(bf,af,det_pos_full)
            plt.subplot(212);plt.plot(times_rppg_full, filt_pos_full, label=f"Filtered POS ({buffers['pos_lowcut_hr']}-{buffers['pos_highcut_hr']}Hz)", color='orange'); plt.title("Filtered POS Signal")
        else:
            plt.subplot(212);plt.plot(times_rppg_full, det_pos_full, label="Detrended POS (Filter Error)", color='red'); plt.title("Detrended POS Signal")
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend(); plt.grid(True)
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig("final_rppg_plot_webcam.png"); plt.show(block=False)    # Respiration Plot - tracking pergerakan bahu
    if len(buffers['respiration_of_raw_signal_list']) > buffers['min_frames_for_resp_output']:
        print("Plotting final Respiration (Shoulder Movement) signals...")
        final_resp_of_full = np.array(list(buffers['respiration_of_raw_signal_list']))
        times_resp_full = np.array(buffers['collected_respiration_of_timestamps'][:len(final_resp_of_full)])
        
        plt.figure(figsize=(15,8))
        plt.suptitle("Respiration from Shoulder Movement (Lucas-Kanade Optical Flow) Analysis", fontsize=14)
        
        # Plot sinyal mentah
        plt.subplot(211)
        plt.plot(times_resp_full, final_resp_of_full, color='r', label="Raw Shoulder Movement")
        plt.title("Raw Respiration Signal from Shoulder Movement")
        plt.xlabel("Time (s)")
        plt.ylabel("Avg. Y Pixel Position")
        plt.legend()
        plt.grid(True)
        
        # Plot sinyal yang difilter
        det_resp_full_of = signal.detrend(final_resp_of_full)
        nyq_r,low_r,high_r = 0.5*fs, buffers['resp_lowcut_hz']/(0.5*fs), buffers['resp_highcut_hz']/(0.5*fs)
        if low_r>0 and high_r<1 and low_r<high_r:
            br,ar = signal.butter(buffers['resp_filter_order'], [low_r,high_r], btype='band')
            filt_resp_full_of = signal.filtfilt(br,ar,det_resp_full_of)
            
            # Deteksi puncak untuk menghitung laju pernapasan rata-rata
            norm_sig = (filt_resp_full_of-np.mean(filt_resp_full_of))/(np.std(filt_resp_full_of)+1e-9)
            peaks, _ = signal.find_peaks(norm_sig, prominence=0.35, distance=fs/(buffers['resp_highcut_hz']*2.5))
            
            plt.subplot(212)
            plt.plot(times_resp_full, filt_resp_full_of, color='g', 
                   label=f"Filtered Shoulder Movement ({buffers['resp_lowcut_hz']}-{buffers['resp_highcut_hz']}Hz)")
            
            # Plot puncak yang terdeteksi
            if len(peaks) > 0:
                plt.plot(times_resp_full[peaks], filt_resp_full_of[peaks], "rx", label="Detected Breaths")
                
                # Hitung dan tampilkan laju pernapasan rata-rata
                if len(peaks) >= 2:
                    avg_rpm = 60.0 / (np.mean(np.diff(peaks)) / fs)
                    plt.title(f"Filtered Respiration Signal - Avg. Rate: {avg_rpm:.1f} breaths/min")
                else:
                    plt.title("Filtered Respiration Signal (Shoulder Movement)")
            else:
                plt.title("Filtered Respiration Signal (Shoulder Movement)")
        else:
            plt.subplot(212)
            plt.plot(times_resp_full, det_resp_full_of, color='m', label="Detrended Resp (Filter Error)")
            plt.title("Detrended Respiration Signal (Shoulder Movement)")
            
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig("final_respiration_shoulder_movement_plot.png")
        plt.show(block=True)

# --- Main Application Orchestrator ---
def run_application(camera_id=2):
    cap, cam_params = initialize_camera_and_parameters(camera_id)
    if not cap: return

    face_mesh, pose_landmarker_obj = initialize_mediapipe(RPPG_MODEL_FILE_NAME)
    if not face_mesh or not pose_landmarker_obj:
        if cap: cap.release()
        return

    buffers_state = initialize_buffers_and_state(cam_params['fs'])
    display_config = setup_display_and_video(cam_params['actual_width'], cam_params['actual_height'], cam_params['fs'])
    
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
                                               (display_config['display_camera_portion_w'], display_config['display_camera_h']))
            
            # Clear graph images
            for g_key in ['rppg_raw_g', 'rppg_filt_g', 'resp_raw_g', 'resp_filt_g']:
                display_config[g_key].fill(255)
                draw_grid(display_config[g_key], display_config['graph_width'], display_config['graph_height'], display_config['grid_color'])

            # Process rPPG
            current_hr, resized_frame_display_after_rppg = process_rppg_frame(
                frame_display_orig_size, face_mesh, buffers_state, cam_params, 
                display_config, current_time_from_start, resized_frame_display
            )
            
            # Process Respiration
            current_rpm, resized_frame_display_after_resp = process_respiration_frame(
                frame_orig_for_pose_of, pose_landmarker_obj, buffers_state, cam_params, 
                display_config, current_time_from_start, frame_count, resized_frame_display_after_rppg
            )
            
            # Assemble and show display
            combined_display_bg = np.ones((display_config['combined_height'], display_config['combined_width'], 3), dtype=np.uint8)*220
            assemble_and_show_display(combined_display_bg, resized_frame_display_after_resp, display_config, current_hr, current_rpm)

            if cv2.waitKey(1)&0xFF==ord('q') or cv2.getWindowProperty(display_config['window_name'],cv2.WND_PROP_VISIBLE)<1: break
    
    except Exception as e:
        print(f"Error in main loop on frame {frame_count}: {e}")
        traceback.print_exc()
        # Ensure values are reset if error occurs mid-loop for final plots
        buffers_state['current_hr'] = 0.0
        buffers_state['current_rpm'] = 0.0
        buffers_state['features_of'] = None 
    finally:
        cleanup_resources(cap, display_config.get('video_writer'), pose_landmarker_obj, face_mesh)
        generate_final_plots(buffers_state, cam_params, display_config)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created '{models_dir}' directory. Please place '{RPPG_MODEL_FILE_NAME}' or 'pose_landmarker_lite.task' there.")
    
    run_application()
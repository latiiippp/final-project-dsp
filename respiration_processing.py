import cv2
import numpy as np
from scipy import signal
import mediapipe_utils
import config

def initialize_shoulder_optical_flow_features(frame_for_of, pose_landmarker_obj, lk_params_dict): # lk_params_dict not used here, uses config
    left_roi, right_roi = mediapipe_utils.get_shoulder_roi_for_optical_flow(frame_for_of, pose_landmarker_obj)
    if left_roi is None and right_roi is None:
        return None, None, (None, None), (False, False)
    
    gray_frame_for_of = cv2.cvtColor(frame_for_of, cv2.COLOR_BGR2GRAY)
    combined_features = []
    valid_shoulders = [False, False]
    
    if left_roi is not None:
        l_x, l_y, r_x, r_y = left_roi
        left_roi_gray = gray_frame_for_of[l_y:r_y, l_x:r_x]
        if left_roi_gray.size > 0 and left_roi_gray.shape[0] >= 5 and left_roi_gray.shape[1] >= 5:
            left_features = cv2.goodFeaturesToTrack(
                left_roi_gray, maxCorners=config.RESP_OF_MAX_CORNERS_PER_SHOULDER, 
                qualityLevel=config.RESP_OF_QUALITY_LEVEL, 
                minDistance=config.RESP_OF_MIN_DISTANCE_FEATURES, 
                blockSize=config.RESP_OF_BLOCK_SIZE_FEATURES
            )
            if left_features is not None:
                left_features = np.float32(left_features)
                left_features[:,:,0] += l_x
                left_features[:,:,1] += l_y
                combined_features.append(left_features)
                valid_shoulders[0] = True
    
    if right_roi is not None:
        l_x, l_y, r_x, r_y = right_roi
        right_roi_gray = gray_frame_for_of[l_y:r_y, l_x:r_x]
        if right_roi_gray.size > 0 and right_roi_gray.shape[0] >= 5 and right_roi_gray.shape[1] >= 5:
            right_features = cv2.goodFeaturesToTrack(
                right_roi_gray, maxCorners=config.RESP_OF_MAX_CORNERS_PER_SHOULDER, 
                qualityLevel=config.RESP_OF_QUALITY_LEVEL, 
                minDistance=config.RESP_OF_MIN_DISTANCE_FEATURES, 
                blockSize=config.RESP_OF_BLOCK_SIZE_FEATURES
            )
            if right_features is not None:
                right_features = np.float32(right_features)
                right_features[:,:,0] += l_x
                right_features[:,:,1] += l_y
                combined_features.append(right_features)
                valid_shoulders[1] = True
    
    if not combined_features:
        return None, None, (left_roi, right_roi), (False, False)
    
    all_features = np.vstack(combined_features)
    return all_features, gray_frame_for_of.copy(), (left_roi, right_roi), tuple(valid_shoulders)

def process_respiration_frame(frame_orig_for_pose_of, pose_landmarker_obj, buffers, cam_params, display_config, current_time_from_start, frame_count, resized_frame_display_in):
    fs, eps = cam_params['fs'], cam_params['eps']
    resized_frame_display = resized_frame_display_in.copy()

    frame_for_of_processing = cv2.resize(frame_orig_for_pose_of, config.STANDARD_SIZE_OF_FOR_OPTICAL_FLOW)
    
    if buffers['features_of'] is None or len(buffers['features_of']) < config.RESP_OF_MIN_FEATURES_FOR_SIGNAL:
        if frame_count > 1:
            cv2.putText(display_config['resp_raw_g'], "Re-Init Shoulders", (10, display_config['graph_height']//2-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)
        
        features, gray_frame, shoulder_rois, valid_shoulders = initialize_shoulder_optical_flow_features(
            frame_for_of_processing, pose_landmarker_obj, buffers['lk_params_of']
        )
        buffers['features_of'] = features
        buffers['old_gray_of'] = gray_frame
        buffers['roi_coords_of'] = shoulder_rois
        buffers['valid_shoulders'] = valid_shoulders
        
        if buffers['features_of'] is None:
            cv2.putText(display_config['resp_raw_g'], "Shoulder Track Failed", (10, display_config['graph_height']//2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)
        # else: # Removed "Shoulder Track OK" to reduce clutter, can be re-added if needed
        #     cv2.putText(display_config['resp_raw_g'], "Shoulder Track OK", (10, display_config['graph_height']//2),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)

    if buffers['features_of'] is not None and buffers['old_gray_of'] is not None:
        frame_gray_of_processing = cv2.cvtColor(frame_for_of_processing, cv2.COLOR_BGR2GRAY)
        new_features, status, error = cv2.calcOpticalFlowPyrLK(buffers['old_gray_of'], frame_gray_of_processing, buffers['features_of'], None, **buffers['lk_params_of'])
        good_new = []
        if new_features is not None and status is not None:
            good_new = new_features[status.flatten() == 1]
        
        left_roi, right_roi = buffers['roi_coords_of'] if isinstance(buffers['roi_coords_of'], tuple) and len(buffers['roi_coords_of']) == 2 else (None, None)
        scale_x_of_to_disp = display_config['display_camera_portion_w'] / config.STANDARD_SIZE_OF_FOR_OPTICAL_FLOW[0]
        scale_y_of_to_disp = display_config['display_camera_h'] / config.STANDARD_SIZE_OF_FOR_OPTICAL_FLOW[1]
        
        if left_roi and buffers['valid_shoulders'][0]:
            l_x, l_y, r_x, r_y = left_roi
            l_d, t_d = int(l_x * scale_x_of_to_disp), int(l_y * scale_y_of_to_disp)
            r_d, b_d = int(r_x * scale_x_of_to_disp), int(r_y * scale_y_of_to_disp)
            cv2.rectangle(resized_frame_display, (display_config['display_camera_portion_w'] - r_d, t_d), (display_config['display_camera_portion_w'] - l_d, b_d), (0, 255, 0), 1)
            cv2.putText(resized_frame_display, "L", (display_config['display_camera_portion_w'] - r_d + 5, t_d + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if right_roi and buffers['valid_shoulders'][1]:
            l_x, l_y, r_x, r_y = right_roi
            l_d, t_d = int(l_x * scale_x_of_to_disp), int(l_y * scale_y_of_to_disp)
            r_d, b_d = int(r_x * scale_x_of_to_disp), int(r_y * scale_y_of_to_disp)
            cv2.rectangle(resized_frame_display, (display_config['display_camera_portion_w'] - r_d, t_d), (display_config['display_camera_portion_w'] - l_d, b_d), (0, 255, 255), 1)
            cv2.putText(resized_frame_display, "R", (display_config['display_camera_portion_w'] - r_d + 5, t_d + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        for pt_new_abs in good_new:
            x_abs_std, y_abs_std = pt_new_abs.ravel()
            disp_x_on_resized = int(x_abs_std * scale_x_of_to_disp)
            disp_y_on_resized = int(y_abs_std * scale_y_of_to_disp)
            cv2.circle(resized_frame_display, (display_config['display_camera_portion_w'] - disp_x_on_resized, disp_y_on_resized), 2, (0, 255, 0), -1)
        
        if len(good_new) > config.RESP_OF_MIN_FEATURES_FOR_SIGNAL:
            avg_y_of = np.mean(good_new[:, 0, 1])
            buffers['respiration_of_raw_signal_list'].append(avg_y_of)
            buffers['collected_respiration_of_timestamps'].append(current_time_from_start)
            buffers['features_of'] = good_new.reshape(-1, 1, 2)
            buffers['old_gray_of'] = frame_gray_of_processing.copy()
        else:
            buffers['features_of'] = None
            if buffers['old_gray_of'] is not None: # Only show if OF was active
                cv2.putText(display_config['resp_raw_g'], "OF Track Lost", (10, display_config['graph_height']//2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, display_config['graph_text_color'], 1)
    
    buffers['current_rpm'] = 0.0
    min_frames_for_resp_output = buffers.get('min_frames_for_resp_output', int(fs*5))

    if len(buffers['respiration_of_raw_signal_list']) >= min_frames_for_resp_output:
        # Use consistent display length, same as rPPG for visual consistency if desired
        # Or use a length specific to respiration, e.g., buffers['respiration_filtered_display_buffer'].maxlen
        display_buffer_len = buffers['pos_display_buffer_size'] 
        resp_seg_of = np.array(list(buffers['respiration_of_raw_signal_list'])[-display_buffer_len:])
        
        if len(resp_seg_of) > int(fs*2):
            det_resp_of = signal.detrend(resp_seg_of)
            
            window_size_med = int(fs * config.RESP_OF_MEDIAN_FILTER_WINDOW_SECONDS)
            if window_size_med % 2 == 0: window_size_med += 1
            if window_size_med < 3: window_size_med = 3
            med_filt_resp = signal.medfilt(det_resp_of, kernel_size=window_size_med)
            
            nyq_r = 0.5*fs
            low_r = config.RESP_OF_LOWCUT_HZ / nyq_r
            high_r = config.RESP_OF_HIGHCUT_HZ / nyq_r
            
            if low_r > 0 and high_r < 1 and low_r < high_r:
                br,ar = signal.butter(config.RESP_OF_FILTER_ORDER, [low_r,high_r], btype='band')
                filt_resp_of = signal.filtfilt(br,ar,med_filt_resp)
                buffers['respiration_filtered_display_buffer'].clear()
                buffers['respiration_filtered_display_buffer'].extend(filt_resp_of)
                
                norm_sig_resp = (filt_resp_of-np.mean(filt_resp_of))/(np.std(filt_resp_of)+eps)
                signal_amplitude_resp = np.abs(norm_sig_resp)
                adaptive_prominence_resp = np.mean(signal_amplitude_resp) * config.RESP_OF_PEAK_PROMINENCE_FACTOR
                prominence_threshold_resp = max(adaptive_prominence_resp, config.RESP_OF_MIN_PEAK_PROMINENCE)
                
                peaks_r,_ = signal.find_peaks(norm_sig_resp, 
                                            prominence=prominence_threshold_resp, 
                                            distance=fs/(config.RESP_OF_HIGHCUT_HZ*2.5),
                                            width=(int(fs*config.RESP_OF_PEAK_WIDTH_MIN_SECONDS), int(fs*config.RESP_OF_PEAK_WIDTH_MAX_SECONDS)))
                
                if len(peaks_r)>=2: 
                    peak_intervals_resp = np.diff(peaks_r)/fs
                    window_size_rpm_avg = min(len(peak_intervals_resp), config.RESP_OF_RPM_MOVING_AVG_INTERVALS)
                    if window_size_rpm_avg > 0:
                        moving_avg_period_resp = np.convolve(peak_intervals_resp, np.ones(window_size_rpm_avg)/window_size_rpm_avg, mode='valid').mean()
                        if moving_avg_period_resp > eps:
                            rpm_raw = 60.0/moving_avg_period_resp
                            buffers['current_rpm'] = np.clip(rpm_raw, config.RESP_MIN_RPM, config.RESP_MAX_RPM)
                
                if len(buffers['respiration_filtered_display_buffer'])>1:
                    data_plot_filt = np.array(list(buffers['respiration_filtered_display_buffer']))
                    min_v,max_v=np.min(data_plot_filt),np.max(data_plot_filt)
                    if max_v-min_v > eps:
                        norm_disp_data=(data_plot_filt-min_v)/(max_v-min_v)
                        num_pts_to_plot = len(norm_disp_data)
                        graph_w = display_config['graph_width']
                        graph_h = display_config['graph_height']
                        if num_pts_to_plot > 1:
                            for i in range(1, num_pts_to_plot):
                                x1 = int((i-1) * (graph_w - 1) / (num_pts_to_plot - 1))
                                x2 = int(i * (graph_w - 1) / (num_pts_to_plot - 1))
                                y1 = graph_h - int(norm_disp_data[i-1] * graph_h)
                                y2 = graph_h - int(norm_disp_data[i] * graph_h)
                                cv2.line(display_config['resp_filt_g'], (x1, y1), (x2, y2), (0,100,0),1)
        
        # Plot raw respiration signal (using resp_seg_of which is already sliced)
        if len(resp_seg_of)>1: # Check if resp_seg_of has enough points
            min_v,max_v=np.min(resp_seg_of),np.max(resp_seg_of)
            if max_v-min_v > eps:
                norm_disp_data=(resp_seg_of-min_v)/(max_v-min_v)
                num_pts_to_plot = len(norm_disp_data)
                graph_w = display_config['graph_width']
                graph_h = display_config['graph_height']
                if num_pts_to_plot > 1:
                    for i in range(1, num_pts_to_plot):
                        x1 = int((i-1) * (graph_w - 1) / (num_pts_to_plot - 1))
                        x2 = int(i * (graph_w - 1) / (num_pts_to_plot - 1))
                        y1 = graph_h - int(norm_disp_data[i-1] * graph_h)
                        y2 = graph_h - int(norm_disp_data[i] * graph_h)
                        cv2.line(display_config['resp_raw_g'], (x1, y1), (x2, y2), (0,0,255),1)
    
    elif buffers['features_of'] is not None : 
        cv2.putText(display_config['resp_raw_g'],"Collecting Shoulder Movement...",(10,display_config['graph_height']//2),cv2.FONT_HERSHEY_SIMPLEX,0.4,display_config['graph_text_color'],1)
    elif buffers['features_of'] is None and buffers['old_gray_of'] is not None: pass # OF was active but lost
    else: 
        cv2.putText(display_config['resp_raw_g'],"Waiting Shoulder Detection...",(10,display_config['graph_height']//2),cv2.FONT_HERSHEY_SIMPLEX,0.4,display_config['graph_text_color'],1)
    
    return buffers['current_rpm'], resized_frame_display
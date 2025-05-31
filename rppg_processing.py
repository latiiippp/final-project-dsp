import cv2
import numpy as np
from scipy import signal
import mediapipe_utils
import utils
import config

def process_rppg_frame(frame_display_orig_size, face_mesh, buffers, cam_params, display_config, current_time_from_start, resized_frame_display_in):
    actual_width, actual_height = cam_params['actual_width'], cam_params['actual_height']
    fs, eps = cam_params['fs'], cam_params['eps']
    resized_frame_display = resized_frame_display_in.copy()

    rgb_frame_mp_face = cv2.cvtColor(frame_display_orig_size, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame_mp_face)
    valid_roi_for_rppg = False

    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        forehead_roi_coords = mediapipe_utils.get_forehead_roi(frame_display_orig_size, face_landmarks, actual_width, actual_height)
        
        if forehead_roi_coords:
            forehead_x_s, forehead_y_s, forehead_x_e, forehead_y_e = forehead_roi_coords
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
    
    buffers['current_hr'] = 0.0
    min_frames_for_pos_output = buffers.get('min_frames_for_pos_output', int(1.6*fs) + int(fs*1))

    if valid_roi_for_rppg and len(buffers['g_signal_list']) >= min_frames_for_pos_output:
        seg_len = min(len(buffers['g_signal_list']), buffers['pos_display_buffer_size'])
        r_s,g_s,b_s = list(buffers['r_signal_list'])[-seg_len:], list(buffers['g_signal_list'])[-seg_len:], list(buffers['b_signal_list'])[-seg_len:]
        
        if len(r_s) > int(1.6*fs): 
            pos_out_full_seg = utils.cpu_POS(np.array([r_s,g_s,b_s]).reshape(1,3,-1), fps=fs)[0]
            buffers['rppg_pos_display_buffer'].clear(); buffers['rppg_pos_display_buffer'].extend(pos_out_full_seg)
            
            if len(buffers['rppg_pos_display_buffer']) > int(fs*2): 
                det_pos = signal.detrend(list(buffers['rppg_pos_display_buffer']))
                nyq = 0.5*fs
                low = config.RPPG_POS_LOWCUT_HZ / nyq
                high = config.RPPG_POS_HIGHCUT_HZ / nyq
                
                if low > 0 and high < 1 and low < high:
                    bf,af = signal.butter(config.RPPG_POS_FILTER_ORDER, [low,high], btype='band')
                    filt_pos = signal.filtfilt(bf,af,det_pos)
                    norm_filt_pos = (filt_pos-np.mean(filt_pos))/(np.std(filt_pos)+eps)
                    peaks,_ = signal.find_peaks(norm_filt_pos, prominence=config.RPPG_PEAK_PROMINENCE, distance=fs/(config.RPPG_POS_HIGHCUT_HZ*2.0)) 
                    if len(peaks)>=2: 
                        buffers['current_hr'] = np.clip(60.0/(np.mean(np.diff(peaks))/fs), config.RPPG_MIN_HR_BPM, config.RPPG_MAX_HR_BPM)
                    
                    min_v,max_v=np.min(filt_pos),np.max(filt_pos)
                    if max_v-min_v > eps:
                        norm_disp_data = (filt_pos-min_v)/(max_v-min_v)
                        num_pts_to_plot = len(norm_disp_data)
                        graph_w = display_config['graph_width']
                        graph_h = display_config['graph_height']
                        if num_pts_to_plot > 1:
                            for i in range(1, num_pts_to_plot):
                                x1 = int((i-1) * (graph_w - 1) / (num_pts_to_plot - 1))
                                x2 = int(i * (graph_w - 1) / (num_pts_to_plot - 1))
                                y1 = graph_h - int(norm_disp_data[i-1] * graph_h)
                                y2 = graph_h - int(norm_disp_data[i] * graph_h)
                                cv2.line(display_config['rppg_filt_g'], (x1, y1), (x2, y2), (0,165,255),1)
        
        if len(buffers['rppg_pos_display_buffer'])>1:
            raw_plot_data = np.array(list(buffers['rppg_pos_display_buffer']))
            min_v,max_v=np.min(raw_plot_data),np.max(raw_plot_data)
            if max_v-min_v > eps:
                norm_disp_data=(raw_plot_data-min_v)/(max_v-min_v)
                num_pts_to_plot = len(norm_disp_data)
                graph_w = display_config['graph_width']
                graph_h = display_config['graph_height']
                if num_pts_to_plot > 1:
                    for i in range(1, num_pts_to_plot):
                        x1 = int((i-1) * (graph_w - 1) / (num_pts_to_plot - 1))
                        x2 = int(i * (graph_w - 1) / (num_pts_to_plot - 1))
                        y1 = graph_h - int(norm_disp_data[i-1] * graph_h)
                        y2 = graph_h - int(norm_disp_data[i] * graph_h)
                        cv2.line(display_config['rppg_raw_g'], (x1, y1), (x2, y2), (255,0,0),1)
    
    elif not valid_roi_for_rppg: 
        cv2.putText(display_config['rppg_raw_g'],"No Face ROI",(10,display_config['graph_height']//2),cv2.FONT_HERSHEY_SIMPLEX,0.4,display_config['graph_text_color'],1)
    else: 
        cv2.putText(display_config['rppg_raw_g'],"Buffering RGB...",(10,display_config['graph_height']//2),cv2.FONT_HERSHEY_SIMPLEX,0.4,display_config['graph_text_color'],1)
    
    return buffers['current_hr'], resized_frame_display
# (This is a new file you would create by refactoring main.py)
import cv2
import mediapipe as mp
import numpy as np
import collections
from scipy import signal
import time

class RPPGProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.buffer_size = 300
        self.rppg_buffer = collections.deque(maxlen=self.buffer_size)
        self.fs = 30  # Default, will be updated
        self.lowcut = 0.1
        self.highcut = 0.5
        self.filter_order = 4
        self.collected_raw_rppg_signal = []
        self.collected_timestamps = []
        self.start_capture_time = 0
        self.is_collecting = False
        self.graph_height = 100 # For mini-graphs
        self.graph_width = 200  # For mini-graphs

    def set_capture_parameters(self, actual_fps, capture_start_time):
        self.fs = actual_fps if actual_fps > 0 else 30
        self.start_capture_time = capture_start_time
        self.is_collecting = True
        self.collected_raw_rppg_signal.clear()
        self.collected_timestamps.clear()
        self.rppg_buffer.clear()

    def process_single_frame(self, bgr_frame, frame_width, frame_height):
        annotated_frame = bgr_frame.copy()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        current_bpm_value = 0.0
        # Initialize empty small images for graphs
        raw_graph_img = np.zeros((self.graph_height, self.graph_width, 3), dtype=np.uint8)
        filtered_graph_img = np.zeros((self.graph_height, self.graph_width, 3), dtype=np.uint8)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            self.mp_drawing.draw_landmarks(
                image=annotated_frame, landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # --- ROI Extraction (simplified from your main.py) ---
            h, w = frame_height, frame_width # Use passed dimensions
            face_pts = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark], dtype=np.int32)
            fx_min, fy_min = np.min(face_pts, axis=0)
            fx_max, fy_max = np.max(face_pts, axis=0)
            face_w, face_h = fx_max - fx_min, fy_max - fy_min

            if face_w > 0 and face_h > 0:
                fh_y_s = fy_min + int(face_h * 0.10)
                fh_y_e = fy_min + int(face_h * 0.25)
                fh_x_s = fx_min + int(face_w * 0.20)
                fh_x_e = fx_min + int(face_w * 0.80)
                fh_roi_orig = bgr_frame[int(fh_y_s):int(fh_y_e), int(fh_x_s):int(fh_x_e)]
                cv2.rectangle(annotated_frame, (int(fh_x_s), int(fh_y_s)), (int(fh_x_e), int(fh_y_e)), (255,0,0), 1)


                if fh_roi_orig.size > 0:
                    g_sig = np.mean(fh_roi_orig[:, :, 1])
                    self.rppg_buffer.append(g_sig)
                    if self.is_collecting:
                        self.collected_raw_rppg_signal.append(g_sig)
                        self.collected_timestamps.append(time.time() - self.start_capture_time)
            # --- End ROI Extraction ---

            # --- Signal Processing & BPM (simplified from your main.py) ---
            if len(self.rppg_buffer) > 1: # Draw raw graph
                # (Full graph drawing logic from main.py would go here, drawing on raw_graph_img)
                cv2.putText(raw_graph_img, "Raw", (10, self.graph_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)


            min_buf_len = int(self.fs * 2)
            if len(self.rppg_buffer) >= min_buf_len:
                current_sig = np.array(self.rppg_buffer)
                detrended = signal.detrend(current_sig)
                if self.fs > 0 and (self.highcut * 2 < self.fs):
                    nyq, low, high = 0.5 * self.fs, self.lowcut / (0.5*self.fs), self.highcut / (0.5*self.fs)
                    if low > 0 and high < 1 and low < high:
                        b, a = signal.butter(self.filter_order, [low, high], btype='band')
                        filtered_sig = signal.filtfilt(b, a, detrended)
                        # (Full graph drawing logic for filtered_sig on filtered_graph_img)
                        cv2.putText(filtered_graph_img, "Filtered", (10, self.graph_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)

                        # BPM Calculation
                        dist_s = int(self.fs * 1.5); prom_v = 0.20 * np.std(filtered_sig) if np.std(filtered_sig) > 1e-6 else None
                        peaks, _ = signal.find_peaks(filtered_sig, distance=dist_s, prominence=prom_v)
                        if len(peaks) >= 2:
                            avg_int_s = np.mean(np.diff(peaks)) / self.fs
                            if avg_int_s > 0: current_bpm_value = np.clip(60.0 / avg_int_s, 5, 35)
            # --- End Signal Processing ---
        else: # No face
            cv2.putText(raw_graph_img, "No Face", (10, self.graph_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
            cv2.putText(filtered_graph_img, "No Face", (10, self.graph_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)


        return annotated_frame, current_bpm_value, raw_graph_img, filtered_graph_img

    def get_collected_data_for_plotting(self):
        return self.collected_raw_rppg_signal, self.collected_timestamps

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()
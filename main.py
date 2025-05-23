import cv2
import mediapipe as mp
import numpy as np
import collections
from scipy import signal
import matplotlib.pyplot as plt # Added import
import time # Added import

def main(camera_id=0):
    """
    Start webcam capture and display in a window with face detection and tracking.
    
    Args:
        camera_id: ID of the camera to use (default is 1)
    """
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)
    
    # Set resolution to 1920x1080 and frame rate to 30 fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get actual resolution (may differ from requested if camera doesn't support it)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam detected. Resolution: {int(actual_width)}x{int(actual_height)} at {int(actual_fps)} fps")
    print("Press 'q' to quit or close the window.")
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Configure face mesh parameters for better performance and accuracy
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,  # For video processing
        max_num_faces=3,          # Track up to 3 faces
        refine_landmarks=True,    # Refine landmarks around eyes and lips
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Drawing specifications for face mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    # Tracking history for stabilization
    landmark_history = []
    history_length = 5  # Number of frames to keep in history
    
    # rPPG signal extraction
    buffer_size = 300  # About 10 seconds at 30 fps
    r_values = collections.deque(maxlen=buffer_size)
    rppg_buffer = collections.deque(maxlen=buffer_size) # Changed from g_values
    b_values = collections.deque(maxlen=buffer_size)
    
    # For displaying the rPPG graph
    graph_width = 400
    graph_height = 200
    raw_graph_img = np.zeros((graph_height, graph_width, 3), dtype=np.uint8) # Renamed for clarity
    filtered_graph_img = np.zeros((graph_height, graph_width, 3), dtype=np.uint8) # For filtered signal

    # Filter parameters
    fs = actual_fps if actual_fps > 0 else 30 # Use actual_fps, default to 30 if not available
    lowcut = 0.1
    highcut = 0.5
    filter_order = 4

    # Data collection for final Matplotlib plot
    collected_raw_rppg_signal = []
    collected_timestamps = []
    start_capture_time = time.time()

    # Video Writer Initialization
    output_video_filename = "rppg_respiration_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    video_writer = None
    if cap.isOpened(): # Ensure cap is open before initializing writer
        # Use actual_width and actual_height obtained from the camera
        frame_width_for_writer = int(actual_width)
        frame_height_for_writer = int(actual_height)
        if frame_width_for_writer > 0 and frame_height_for_writer > 0 and fs > 0:
            video_writer = cv2.VideoWriter(output_video_filename, fourcc, float(fs), (frame_width_for_writer, frame_height_for_writer))
            print(f"Output video will be saved to: {output_video_filename}")
        else:
            print("Warning: Could not initialize video writer due to invalid frame dimensions or FPS.")
            print(f"Frame Width: {frame_width_for_writer}, Frame Height: {frame_height_for_writer}, FPS: {fs}")
    
    # Create window
    window_name = 'Webcam'
    cv2.namedWindow(window_name)
    
    # Main loop
    while True:
        try:
            # Get and process frame
            ret, frame = cap.read()
            
            if frame is None:
                print("Failed to grab frame")
                break
                
            # Mirror the frame horizontally (flip around y-axis)
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe Face Mesh
            results = face_mesh.process(rgb_frame)
            
            # If faces detected, draw landmarks and track faces
            if results.multi_face_landmarks:
                current_landmarks = []
                
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Extract key landmarks for tracking
                    face_points = np.array([
                        [int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])]
                        for lm in face_landmarks.landmark
                    ])
                    current_landmarks.append(face_points)
                    
                    # Draw face mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Draw face contours
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Use all landmarks to create face ROI that matches the face mesh
                    face_boundary_points = np.array([
                        [int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])]
                        for lm in face_landmarks.landmark
                    ])
                    
                    # Get the bounding box
                    face_x_min = np.min(face_boundary_points[:, 0])
                    face_y_min = np.min(face_boundary_points[:, 1])
                    face_x_max = np.max(face_boundary_points[:, 0])
                    face_y_max = np.max(face_boundary_points[:, 1])
                    
                    # Add minimal padding (2% of face size)
                    face_width = face_x_max - face_x_min
                    face_height = face_y_max - face_y_min
                    padding_x = int(face_width * 0.02)
                    padding_y = int(face_height * 0.02)

                    # Apply padding with boundary check
                    face_x_min = max(0, face_x_min - padding_x)
                    face_y_min = max(0, face_y_min - padding_y)
                    face_x_max = min(frame.shape[1], face_x_max + padding_x)
                    face_y_max = min(frame.shape[0], face_y_max + padding_y)
                    
                    # Draw face ROI rectangle
                    cv2.rectangle(frame, (face_x_min, face_y_min), (face_x_max, face_y_max), 
                                 (255, 255, 0), 2)
                    cv2.putText(frame, "Face ROI (MediaPipe)", (face_x_min, face_y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # Define and draw forehead ROI based on MediaPipe face bounding box
                    forehead_mp_y_start = face_y_min + int(face_height * 0.10)
                    forehead_mp_y_end = face_y_min + int(face_height * 0.25)
                    forehead_mp_x_start = face_x_min + int(face_width * 0.20) 
                    forehead_mp_x_end = face_x_min + int(face_width * 0.80)   
                    
                    forehead_mp_y_start, forehead_mp_y_end = int(forehead_mp_y_start), int(forehead_mp_y_end)
                    forehead_mp_x_start, forehead_mp_x_end = int(forehead_mp_x_start), int(forehead_mp_x_end)

                    cv2.rectangle(frame, (forehead_mp_x_start, forehead_mp_y_start), (forehead_mp_x_end, forehead_mp_y_end), (255, 0, 0), 2) # Blue color
                    cv2.putText(frame, "Forehead ROI (MP)", (forehead_mp_x_start, forehead_mp_y_start - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # Extract face ROI and forehead ROI for rPPG
                    # The full face_roi (MediaPipe) is already defined as face_roi = frame[face_y_min:face_y_max, face_x_min:face_x_max]
                    # We will use this as a fallback. Let's get the image data for it.
                    face_roi_img_data = frame[face_y_min:face_y_max, face_x_min:face_x_max]
                    forehead_roi_img_data = frame[forehead_mp_y_start:forehead_mp_y_end, forehead_mp_x_start:forehead_mp_x_end]
                    
                    # Prioritize forehead ROI for rPPG signal extraction
                    r_signal_source, g_signal_source, b_signal_source = 0, 0, 0
                    valid_roi_for_signal = False

                    if forehead_roi_img_data.size > 0:
                        r_signal_source = np.mean(forehead_roi_img_data[:, :, 2])
                        g_signal_source = np.mean(forehead_roi_img_data[:, :, 1])
                        b_signal_source = np.mean(forehead_roi_img_data[:, :, 0])
                        valid_roi_for_signal = True
                    elif face_roi_img_data.size > 0: # Fallback to full face ROI
                        r_signal_source = np.mean(face_roi_img_data[:, :, 2])
                        g_signal_source = np.mean(face_roi_img_data[:, :, 1])
                        b_signal_source = np.mean(face_roi_img_data[:, :, 0])
                        valid_roi_for_signal = True

                    if valid_roi_for_signal:
                        r_values.append(r_signal_source)
                        rppg_buffer.append(g_signal_source) # Changed from g_values
                        b_values.append(b_signal_source)

                        # Collect data for final plot
                        collected_raw_rppg_signal.append(g_signal_source)
                        collected_timestamps.append(time.time() - start_capture_time)
                        
                        cv2.putText(frame, f"RGB (Signal Src): {r_signal_source:.0f}, {g_signal_source:.0f}, {b_signal_source:.0f}", 
                                   (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Process signal for raw graph
                        if len(rppg_buffer) > 10: 
                            raw_signal_for_graph = np.array(rppg_buffer) 
                            if raw_signal_for_graph.size > 0:
                                min_val = np.min(raw_signal_for_graph)
                                max_val = np.max(raw_signal_for_graph)
                                if max_val - min_val > 0:
                                    raw_normalized = (raw_signal_for_graph - min_val) / (max_val - min_val)
                                    
                                    raw_graph_img.fill(0)  
                                    plot_points = min(graph_width, len(raw_normalized))
                                    for i in range(1, plot_points):
                                        pt1 = (i-1, graph_height - int(raw_normalized[-plot_points+i-1] * graph_height))
                                        pt2 = (i, graph_height - int(raw_normalized[-plot_points+i] * graph_height))
                                        cv2.line(raw_graph_img, pt1, pt2, (0, 255, 0), 2) # Green for raw

                        # Process and display filtered rPPG signal
                        # Need enough data points for filtering, e.g., more than 2 * filter_order
                        # and also to avoid issues with very short signals.
                        # A common rule of thumb is at least 3 times the filter order, or a few seconds of data.
                        # Let's use a practical minimum like 1 second of data if available.
                        min_buffer_len_for_filter = int(fs) # Require at least 1 second of data
                        if len(rppg_buffer) >= min_buffer_len_for_filter:
                            current_rppg_signal = np.array(rppg_buffer)
                            
                            # 1. Detrend
                            detrended_signal = signal.detrend(current_rppg_signal)
                            
                            # 2. Butterworth Bandpass Filter
                            # Ensure fs is valid for filter design
                            if fs > 0 and (highcut * 2 < fs): # Nyquist criterion
                                nyquist_freq = 0.5 * fs
                                low = lowcut / nyquist_freq
                                high = highcut / nyquist_freq
                                
                                # Check if normalized frequencies are valid
                                if low > 0 and high < 1 and low < high:
                                    b, a = signal.butter(filter_order, [low, high], btype='band')
                                    filtered_rppg = signal.filtfilt(b, a, detrended_signal)

                                    # Normalize filtered signal for plotting
                                    if filtered_rppg.size > 0:
                                        min_filt_val = np.min(filtered_rppg)
                                        max_filt_val = np.max(filtered_rppg)
                                        if max_filt_val - min_filt_val > 0:
                                            filtered_normalized = (filtered_rppg - min_filt_val) / (max_filt_val - min_filt_val)
                                            
                                            filtered_graph_img.fill(0)
                                            plot_points_filt = min(graph_width, len(filtered_normalized))
                                            for i in range(1, plot_points_filt):
                                                pt1_filt = (i-1, graph_height - int(filtered_normalized[-plot_points_filt+i-1] * graph_height))
                                                pt2_filt = (i, graph_height - int(filtered_normalized[-plot_points_filt+i] * graph_height))
                                                cv2.line(filtered_graph_img, pt1_filt, pt2_filt, (0, 255, 255), 2) # Yellow for filtered
                                else:
                                    # Draw placeholder or error on filtered_graph_img if filter params invalid
                                    cv2.putText(filtered_graph_img, "Filter Error", (10, graph_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                            else:
                                # Draw placeholder or error on filtered_graph_img if fs is invalid
                                cv2.putText(filtered_graph_img, "FS Error", (10, graph_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        else:
                            # Not enough data for filtering yet, clear or show message
                            filtered_graph_img.fill(0)
                            cv2.putText(filtered_graph_img, "Buffering...", (10, graph_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1)


                    # Display the raw signal graph
                    if 'raw_graph_img' in locals():
                        h_raw, w_raw = raw_graph_img.shape[:2]
                        frame[20:20+h_raw, frame.shape[1]-20-w_raw:frame.shape[1]-20] = raw_graph_img
                        cv2.putText(frame, "Raw rPPG (Green Channel)", (frame.shape[1]-20-w_raw, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


                    # Display the filtered signal graph (below the raw signal graph)
                    if 'filtered_graph_img' in locals():
                        h_filt, w_filt = filtered_graph_img.shape[:2]
                        # Position it below the raw graph, with some padding
                        y_offset_for_filt_graph = 20 + h_raw + 10 # 10px padding
                        if y_offset_for_filt_graph + h_filt < frame.shape[0]: # Check if it fits
                             frame[y_offset_for_filt_graph : y_offset_for_filt_graph + h_filt, frame.shape[1]-20-w_filt : frame.shape[1]-20] = filtered_graph_img
                             cv2.putText(frame, "Filtered rPPG", (frame.shape[1]-20-w_filt, y_offset_for_filt_graph - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    
                # Update landmark history
                landmark_history.append(current_landmarks)
                if len(landmark_history) > history_length:
                    landmark_history.pop(0)
                
                # Display tracking info
                cv2.putText(frame, f"Tracking {len(results.multi_face_landmarks)} face(s)", 
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            else:
                # No faces detected
                cv2.putText(frame, "No faces detected", 
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow(window_name, frame)

            # Write frame to video file
            if video_writer is not None:
                video_writer.write(frame)
            
            # Check if window was closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Release resources
    face_mesh.close()
    cap.release()
    if video_writer is not None:
        video_writer.release()
    
    # Generate Matplotlib plot for collected signals
    if collected_raw_rppg_signal and len(collected_raw_rppg_signal) > 2 * filter_order + 1:
        print("Generating Matplotlib plot of rPPG signals...")
        final_raw_signal = np.array(collected_raw_rppg_signal)
        final_time_axis = np.array(collected_timestamps)

        # Ensure fs is valid for filter design for the plot
        plot_fs = fs # Use the same fs as in the main loop
        if plot_fs <= 0:
            print("Warning: Invalid sampling frequency for plotting. Using default of 30 Hz.")
            plot_fs = 30
        
        # 1. Detrend for plot
        detrended_plot_signal = signal.detrend(final_raw_signal)
        
        # 2. Butterworth Bandpass Filter for plot
        filtered_plot_signal = None
        if plot_fs > 0 and (highcut * 2 < plot_fs): # Nyquist criterion
            nyquist_plot = 0.5 * plot_fs
            low_plot = lowcut / nyquist_plot
            high_plot = highcut / nyquist_plot
            
            if low_plot > 0 and high_plot < 1 and low_plot < high_plot:
                b_plot, a_plot = signal.butter(filter_order, [low_plot, high_plot], btype='band')
                filtered_plot_signal = signal.filtfilt(b_plot, a_plot, detrended_plot_signal)
            else:
                print("Warning: Could not design filter for Matplotlib plot due to invalid normalized frequencies.")
        else:
            print("Warning: Could not design filter for Matplotlib plot due to invalid sampling frequency or cutoffs.")

        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(final_time_axis, final_raw_signal, label='Raw rPPG Signal (Green Channel)')
        plt.title('Raw rPPG Signal Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        if filtered_plot_signal is not None:
            plt.plot(final_time_axis, filtered_plot_signal, label=f'Filtered rPPG Signal ({lowcut}-{highcut} Hz)', color='orange')
        else:
            plt.plot(final_time_axis, detrended_plot_signal, label='Detrended rPPG Signal (Filter Failed)', color='red') # Plot detrended if filter failed
        plt.title('Processed rPPG Signal Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        try:
            plt.savefig("rppg_signals_plot.png")
            print(f"Matplotlib plot saved to rppg_signals_plot.png")
        except Exception as e:
            print(f"Error saving Matplotlib plot: {e}")
        plt.show()
    else:
        print("Not enough data collected to generate Matplotlib plot or filter requirements not met.")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
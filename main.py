import cv2
import mediapipe as mp
import numpy as np
import collections

def main(camera_id=1):
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
    g_values = collections.deque(maxlen=buffer_size)
    b_values = collections.deque(maxlen=buffer_size)
    
    # For displaying the rPPG graph
    graph_width = 400
    graph_height = 200
    graph_img = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
    
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
                    cv2.putText(frame, "Face ROI", (face_x_min, face_y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Extract face ROI
                    face_roi = frame[face_y_min:face_y_max, face_x_min:face_x_max]
                    
                    # Only process if the ROI is valid
                    if face_roi.size > 0:
                        # Calculate mean RGB values from face ROI
                        face_r = np.mean(face_roi[:, :, 2])
                        face_g = np.mean(face_roi[:, :, 1])
                        face_b = np.mean(face_roi[:, :, 0])
                        
                        # Add face ROI values to buffers
                        r_values.append(face_r)
                        g_values.append(face_g)
                        b_values.append(face_b)
                        
                        # Display RGB values
                        cv2.putText(frame, f"RGB: {face_r:.0f}, {face_g:.0f}, {face_b:.0f}", 
                                   (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Process signal for graph
                        if len(g_values) > 10:
                            g_signal = np.array(g_values)
                            if np.max(g_signal) - np.min(g_signal) > 0:
                                g_normalized = (g_signal - np.min(g_signal)) / (np.max(g_signal) - np.min(g_signal))
                                
                                # Create graph
                                graph_img.fill(0)  # Clear previous graph
                                plot_points = min(graph_width, len(g_normalized))
                                for i in range(1, plot_points):
                                    pt1 = (i-1, graph_height - int(g_normalized[-plot_points+i-1] * graph_height))
                                    pt2 = (i, graph_height - int(g_normalized[-plot_points+i] * graph_height))
                                    cv2.line(graph_img, pt1, pt2, (0, 255, 0), 2)
                                
                    # Display the graph
                    if 'graph_img' in locals():
                        h, w = graph_img.shape[:2]
                        frame[20:20+h, frame.shape[1]-20-w:frame.shape[1]-20] = graph_img
                    
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
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
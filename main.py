import cv2
import mediapipe as mp

def start_webcam(camera_id=0):
    """
    Start webcam capture with face tracking using MediaPipe.
    
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
    
    # Get actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam detected. Resolution: {int(actual_width)}x{int(actual_height)} at {int(actual_fps)} fps")
    print("Press 'q' to quit or close the window.")
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Create window
    window_name = 'Webcam Face Tracking'
    cv2.namedWindow(window_name)
    
    # Main loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Failed to capture image.")
            break
            
        # Mirror the frame horizontally (flip around y-axis)
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect faces
        results = face_detection.process(rgb_frame)
        
        # Draw face detections
        if results.detections:
            for detection in results.detections:
                # Draw detection box
                mp_drawing.draw_detection(frame, detection)
                
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Draw additional information
                cv2.putText(frame, f"Face detected", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    face_detection.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

# Call the function to start webcam
if __name__ == "__main__":
    start_webcam()
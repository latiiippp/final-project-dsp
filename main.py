import cv2

def start_webcam(camera_id=1):
    """
    Start webcam capture and display in a window.
    
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
    
    # Create window
    window_name = 'Webcam'
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
            
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

# Call the function to start webcam
if __name__ == "__main__":
    start_webcam()
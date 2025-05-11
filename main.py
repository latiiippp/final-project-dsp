import cv2

def start_webcam(camera_id=1):
    """
    Start webcam capture and display in a window.
    
    Args:
        camera_id: ID of the camera to use (default is 1)
    """
    # Flag to track if window should close
    window_closed = False
    
    # Callback function for window events
    def on_window_close(event, x, y, flags, param):
        nonlocal window_closed
        if event == cv2.EVENT_WINDOWCLOSED:
            window_closed = True
    
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
    
    # Set mouse callback to detect window close
    cv2.setMouseCallback(window_name, on_window_close)
    
    # Main loop
    while not window_closed:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Failed to capture image.")
            break
            
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Alternative way to check if window was closed
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
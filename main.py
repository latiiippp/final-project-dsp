import cv2
import numpy as np
import mediapipe as mp

class WebcamStream:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        # Set resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure Face Mesh parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_face_mesh(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect faces
        results = self.face_mesh.process(rgb_frame)
        
        # Create a copy for visualization
        annotated_frame = frame.copy()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Extract key landmarks (example: eyes, nose, mouth)
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
        
        return annotated_frame, landmarks if results.multi_face_landmarks else (annotated_frame, None)
    
    def preprocess_frame(self, frame):
        # Basic preprocessing
        frame = cv2.resize(frame, (640, 480))
        
        # Process face mesh
        mesh_frame, landmarks = self.process_face_mesh(frame)
        
        return {
            'original': frame,
            'face_mesh': mesh_frame,
            'landmarks': landmarks
        }
    
    def read(self):
        ret, frame = self.stream.read()
        if not ret:
            return None
        
        return self.preprocess_frame(frame)
    
    def release(self):
        self.stream.release()
        self.face_mesh.close()

def main():
    # Initialize webcam stream
    webcam = WebcamStream(src=0, width=640, height=480)
    
    while True:
        try:
            # Get and process frame
            frames = webcam.read()
            
            if frames is None:
                print("Failed to grab frame")
                break
            
            # Display processed frames
            cv2.imshow('Original', frames['original'])
            cv2.imshow('Face Mesh', frames['face_mesh'])
            
            # Print landmark coordinates if detected
            if frames['landmarks']:
                # Example: Print nose tip coordinates (landmark 4)
                nose_tip = frames['landmarks'][4]
                print(f"Nose tip - x: {nose_tip['x']:.2f}, y: {nose_tip['y']:.2f}, z: {nose_tip['z']:.2f}")
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Clean up
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
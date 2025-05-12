import cv2
import threading
import queue
import time

class WebcamStream:
    def __init__(self, src=0, buffer_size=3):
        self.stream = cv2.VideoCapture(src)
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        
    def start(self):
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            
            if not self.buffer.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
                    return
                
                self.buffer.put(frame)
    
    def read(self):
        return self.buffer.get()
    
    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    # Initialize webcam stream
    webcam = WebcamStream(src=0).start()
    time.sleep(1.0)  # Allow camera to warm up
    
    while True:
        try:
            # Get frame from buffer
            frame = webcam.read()
            
            # Display the frame
            cv2.imshow('Webcam Feed', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error: {e}")
            break
    
    # Clean up
    webcam.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
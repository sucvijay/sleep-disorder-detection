import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time

class DrowsinessDetector:
    def __init__(self):
        # Facial landmark predictor and detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Eye landmark indices
        (self.lStart, self.lEnd) = (42, 48)
        (self.rStart, self.rEnd) = (36, 42)
        
        # Drowsiness detection parameters
        self.EYE_AR_THRESH = 0.2  # Eye Aspect Ratio threshold
        self.EYE_AR_CONSEC_FRAMES = 16  # Frames to consider as drowsy
        self.YAWN_THRESH = 20  # Yawn threshold
        
        # Tracking variables
        self.COUNTER = 0
        self.TOTAL_BLINKS = 0
        self.start_time = time.time()

    def eye_aspect_ratio(self, eye):
        """
        Calculate Eye Aspect Ratio (EAR)
        """
        # Vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Horizontal eye landmark
        C = dist.euclidean(eye[0], eye[3])
        
        # EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        """
        Calculate Mouth Aspect Ratio (MAR)
        """
        # Vertical mouth landmarks
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        
        # Horizontal mouth landmark
        D = dist.euclidean(mouth[12], mouth[16])
        
        # MAR
        mar = (A + B + C) / (3 * D)
        return mar

    def detect_drowsiness(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        rects = self.detector(gray, 0)
        
        # Flag to check if drowsy
        is_drowsy = False
        
        for rect in rects:
            # Get facial landmarks
            shape = self.predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Extract eye and mouth landmarks
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth = shape[48:68]
            
            # Calculate aspect ratios
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # Calculate mouth aspect ratio
            mar = self.mouth_aspect_ratio(mouth)
            
            # Draw eye contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            
            # Drowsiness detection logic
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    is_drowsy = True
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.COUNTER = 0
            
            # Yawning detection (additional sign of fatigue)
            if mar > self.YAWN_THRESH:
                cv2.putText(frame, "YAWNING!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                is_drowsy = True
            
            # Display metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, is_drowsy

def main():
    # Initialize drowsiness detector
    drowsy_detector = DrowsinessDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect drowsiness
        frame, is_drowsy = drowsy_detector.detect_drowsiness(frame)
        
        # Show the frame
        cv2.imshow("Drowsiness Detection", frame)
        
        # Break loop on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
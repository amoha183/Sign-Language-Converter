"""
Hand Gesture Prediction and Real-Time Recognition

This script defines the HandGesturePredictor class for real-time hand gesture recognition using a webcam.
It uses MediaPipe for hand landmark detection and a trained Keras model for sign language letter prediction.
Includes visualization of hand skeleton and prediction overlays.
"""
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model

class HandGesturePredictor:
    def __init__(self):
        # Load the trained model for sign language recognition
        self.model = load_model('checkpoint_sign_converter_alkadya.keras', compile=False)
        self.letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Initialize MediaPipe Hands with improved parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lowered for better detection
            min_tracking_confidence=0.3,   # Lowered for better tracking
            model_complexity=1             # Using complex model for better accuracy
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Custom drawing specifications for better visibility
        self.drawing_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0),  # Green color
            thickness=2,
            circle_radius=2
        )
        self.connection_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0),  # Green color
            thickness=2,
            circle_radius=2
        )
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Set window size for webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Define the fixed region for hand detection (400x400)
        self.detection_size = 400
        self.padding = 50  # Padding around the detection region
        
        # Initialize variables for smoothing
        self.prev_landmarks = None
        self.smoothing_factor = 0.5
        
    def get_detection_region(self, frame):
        # Calculate the center position for the detection region
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        # Calculate the region coordinates
        x1 = center_x - self.detection_size // 2
        y1 = center_y - self.detection_size // 2
        x2 = x1 + self.detection_size
        y2 = y1 + self.detection_size
        
        return (x1, y1, x2, y2)
    
    def smooth_landmarks(self, current_landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return current_landmarks
        
        # Smooth each landmark position
        smoothed_landmarks = []
        for curr, prev in zip(current_landmarks, self.prev_landmarks):
            x = int(curr[0] * (1 - self.smoothing_factor) + prev[0] * self.smoothing_factor)
            y = int(curr[1] * (1 - self.smoothing_factor) + prev[1] * self.smoothing_factor)
            smoothed_landmarks.append((x, y))
        
        self.prev_landmarks = smoothed_landmarks
        return smoothed_landmarks
    
    def detect_hand_landmarks(self, frame, region):
        x1, y1, x2, y2 = region
        
        # Extract the region of interest
        roi = frame[y1:y2, x1:x2]
        
        # Convert to RGB for MediaPipe
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_roi)
        
        if results.multi_hand_landmarks:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Convert landmarks to pixel coordinates
            h, w = roi.shape[:2]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            # Apply smoothing to landmarks
            smoothed_landmarks = self.smooth_landmarks(landmarks)
            
            # Get bounding box of landmarks
            x_coords = [x for x, _ in smoothed_landmarks]
            y_coords = [y for _, y in smoothed_landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            x_min = max(0, x_min - self.padding)
            y_min = max(0, y_min - self.padding)
            x_max = min(w, x_max + self.padding)
            y_max = min(h, y_max + self.padding)
            
            # Convert back to original frame coordinates
            x_min += x1
            y_min += y1
            x_max += x1
            y_max += y1
            
            return hand_landmarks, (x_min, y_min, x_max - x_min, y_max - y_min)
        
        self.prev_landmarks = None
        return None, None
    
    def create_skeleton_image(self, hand_landmarks):
        # Create a white background
        skeleton_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        # Draw the hand skeleton in green
        self.mp_draw.draw_landmarks(
            skeleton_img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.drawing_spec, self.connection_spec)
        # Normalize for model
        skeleton_img = skeleton_img / 255.0
        return skeleton_img
    
    def predict_letter(self, hand_image):
        # Add batch dimension
        hand_image = np.expand_dims(hand_image, axis=0)
        
        # Get prediction
        predictions = self.model.predict(hand_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        predicted_letter = self.letters[predicted_class]
        
        return predicted_letter, confidence
    
    def draw_guides(self, frame, region):
        x1, y1, x2, y2 = region
        
        # Draw the main detection box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw crosshair in the center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        crosshair_size = 20
        cv2.line(frame, (center_x - crosshair_size, center_y),
                (center_x + crosshair_size, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - crosshair_size),
                (center_x, center_y + crosshair_size), (0, 255, 0), 2)
        
        # Draw corner guides
        guide_size = 30
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + guide_size, y1), (0, 255, 0), 2)
        cv2.line(frame, (x1, y1), (x1, y1 + guide_size), (0, 255, 0), 2)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - guide_size, y1), (0, 255, 0), 2)
        cv2.line(frame, (x2, y1), (x2, y1 + guide_size), (0, 255, 0), 2)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + guide_size, y2), (0, 255, 0), 2)
        cv2.line(frame, (x1, y2), (x1, y2 - guide_size), (0, 255, 0), 2)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - guide_size, y2), (0, 255, 0), 2)
        cv2.line(frame, (x2, y2), (x2, y2 - guide_size), (0, 255, 0), 2)
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Get the detection region
            detection_region = self.get_detection_region(frame)
            
            # Draw guides
            self.draw_guides(frame, detection_region)
            
            # Detect hand landmarks
            hand_landmarks, hand_region = self.detect_hand_landmarks(frame, detection_region)
            
            if hand_landmarks and hand_region:
                x, y, w, h = hand_region
                
                # Create a white skeleton image for prediction
                skeleton_img = self.create_skeleton_image(hand_landmarks)
                predicted_letter, confidence = self.predict_letter(skeleton_img)
                # Show the skeleton image in a separate window
                display_img = (skeleton_img * 255).astype(np.uint8)
                cv2.putText(display_img, f"{predicted_letter} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Hand Skeleton (Model Input)', display_img)
                # Draw prediction on main frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})",
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (0, 255, 0), 2)
            else:
                # Display message when no hand is detected
                cv2.putText(frame, "No hand detected", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Show a blank skeleton window
                blank = np.ones((400, 400, 3), dtype=np.uint8) * 255
                cv2.imshow('Hand Skeleton (Model Input)', blank)
            
            # Display instructions
            cv2.putText(frame, "Position your hand in the green box", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up   
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    predictor = HandGesturePredictor()
    predictor.run() 
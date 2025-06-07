"""
Sign Language To Text Conversion GUI

This script provides a user-friendly Tkinter-based GUI for testing a hand gesture AI model.
Features:
- Live webcam feed and hand skeleton visualization
- Automatic and manual character addition
- Sentence builder with Add, Delete, Space, and Clear buttons
- Word suggestion/completion for the last word
- Uses the HandGesturePredictor class for AI predictions
"""
import tkinter as tk
from tkinter import Label, Button, Frame, StringVar
from PIL import Image, ImageTk
import cv2
import numpy as np
from realtime_predict import HandGesturePredictor
import threading
import time

class SignLanguageGUI:
    def __init__(self, root):
        # Initialize the main window and all variables/components
        self.root = root
        self.root.title("Sign Language To Text Conversion (GUI Test)")
        self.root.geometry("1100x700")
        self.root.resizable(False, False)

        # Predictor
        self.predictor = HandGesturePredictor()  # Hand gesture AI model
        self.cap = self.predictor.cap  # Webcam capture

        # Variables
        self.current_char = StringVar(value="...")  # Current predicted character
        self.current_conf = StringVar(value="...")  # Current prediction confidence
        self.sentence = StringVar(value="")  # The sentence being built

        # For auto-add
        self.last_predicted = None  # Last predicted character
        self.last_predicted_time = None  # Time when last character appeared
        self.auto_add_delay = 3  # seconds to wait before auto-adding
        self.added_recently = False  # Prevents repeated auto-adds

        # Layout
        self.setup_layout()  # Build the GUI layout

        # For stopping the thread
        self.running = True
        self.update_thread = threading.Thread(target=self.update_frames)  # Thread for video updates
        self.update_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle window close

    def setup_layout(self):
        # Set up all GUI widgets and layout
        # Title
        Label(self.root, text="Sign Language To Text Conversion", font=("Courier", 18, "bold")).pack(pady=10)

        # Frames for video and skeleton
        frame = Frame(self.root)
        frame.pack(pady=10)
        self.video_label = Label(frame)
        self.video_label.grid(row=0, column=0, padx=20)
        self.skeleton_label = Label(frame)
        self.skeleton_label.grid(row=0, column=1, padx=20)

        # Prediction info
        Label(self.root, text="Character:", font=("Courier", 12, "bold")).pack()
        Label(self.root, textvariable=self.current_char, font=("Courier", 12)).pack()
        Label(self.root, text="Confidence:", font=("Courier", 12, "bold")).pack()
        Label(self.root, textvariable=self.current_conf, font=("Courier", 12)).pack()

        # Sentence
        Label(self.root, text="Sentence:", font=("Courier", 12, "bold")).pack(pady=(10,0))
        Label(self.root, textvariable=self.sentence, font=("Courier", 12)).pack()

        # Suggestions
        self.suggestion_frame = Frame(self.root)
        self.suggestion_frame.pack(pady=(0, 10))
        self.suggestion_buttons = []  # List of suggestion button widgets
        self.word_list = [
            'HELLO', 'HELP', 'HOUSE', 'HAPPY', 'HOW', 'HAVE', 'HE', 'HER', 'HIM', 'HIS',
            'WORLD', 'WHERE', 'WHAT', 'WHEN', 'WHO', 'WHY', 'YES', 'NO', 'NEXT', 'NAME',
            'SIGN', 'LANGUAGE', 'TEXT', 'CONVERT', 'CONVERSION', 'CHARACTER', 'SENTENCE',
            'GOOD', 'BAD', 'PLEASE', 'THANK', 'YOU', 'YOUR', 'MY', 'MINE', 'OUR', 'THEIR',
            'IS', 'ARE', 'AM', 'CAN', 'DO', 'SEE', 'SAY', 'GO', 'COME', 'MAKE', 'TAKE',
            'GIVE', 'GET', 'WANT', 'LIKE', 'LOVE', 'EAT', 'DRINK', 'WATER', 'FOOD', 'FIND'
        ]
        self.update_suggestions()

        # Buttons
        btn_frame = Frame(self.root)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Add", command=self.add_char, width=10).grid(row=0, column=0, padx=5)
        Button(btn_frame, text="Delete", command=self.delete_char, width=10).grid(row=0, column=1, padx=5)
        Button(btn_frame, text="Space", command=self.add_space, width=10).grid(row=0, column=2, padx=5)
        Button(btn_frame, text="Clear", command=self.clear_sentence, width=10).grid(row=0, column=3, padx=5)

    def update_frames(self):
        # Main loop for updating webcam and prediction results
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            detection_region = self.predictor.get_detection_region(frame)
            self.predictor.draw_guides(frame, detection_region)
            hand_landmarks, hand_region = self.predictor.detect_hand_landmarks(frame, detection_region)

            # Prepare skeleton image
            if hand_landmarks and hand_region:
                # If a hand is detected, predict and display results
                skeleton_img = self.predictor.create_skeleton_image(hand_landmarks)
                predicted_letter, confidence = self.predictor.predict_letter(skeleton_img)
                self.current_char.set(predicted_letter)
                self.current_conf.set(f"{confidence:.2f}")
                display_img = (skeleton_img * 255).astype(np.uint8)

                # Auto-add logic
                now = time.time()
                if predicted_letter == self.last_predicted:
                    if self.last_predicted_time is not None and not self.added_recently:
                        if now - self.last_predicted_time >= self.auto_add_delay:
                            # Add character if held for 3 seconds
                            self.sentence.set(self.sentence.get() + predicted_letter)
                            self.added_recently = True
                            self.update_suggestions()
                    # else: still waiting
                else:
                    # New character detected, reset timer
                    self.last_predicted = predicted_letter
                    self.last_predicted_time = now
                    self.added_recently = False
            else:
                # No hand detected, reset prediction and timer
                self.current_char.set("...")
                self.current_conf.set("...")
                display_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
                self.last_predicted = None
                self.last_predicted_time = None
                self.added_recently = False

            # Draw detection box on main frame
            x1, y1, x2, y2 = detection_region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Convert images for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_pil = img_pil.resize((400, 300))
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            skeleton_pil = Image.fromarray(display_img.astype(np.uint8))
            skeleton_pil = skeleton_pil.resize((300, 300))
            skeleton_imgtk = ImageTk.PhotoImage(image=skeleton_pil)
            self.skeleton_label.imgtk = skeleton_imgtk
            self.skeleton_label.configure(image=skeleton_imgtk)

            # Update every 30 ms
            self.root.after(30)

    def add_char(self):
        # Manually add the current character to the sentence
        char = self.current_char.get()
        if char and char != "...":
            self.sentence.set(self.sentence.get() + char)
            self.update_suggestions()

    def delete_char(self):
        # Delete the last character from the sentence
        current = self.sentence.get()
        if current:
            self.sentence.set(current[:-1])
            self.hide_suggestions()

    def clear_sentence(self):
        # Clear the entire sentence
        self.sentence.set("")
        self.hide_suggestions()

    def hide_suggestions(self):
        # Remove all suggestion buttons
        for btn in self.suggestion_buttons:
            btn.destroy()
        self.suggestion_buttons = []

    def update_suggestions(self):
        # Update the suggestion buttons based on the last word
        # Remove old suggestion buttons
        for btn in self.suggestion_buttons:
            btn.destroy()
        self.suggestion_buttons = []
        # Get the last word being typed
        sentence = self.sentence.get()
        words = sentence.split()
        if not words:
            return
        last_word = words[-1].upper()
        if not last_word:
            return
        # Find suggestions
        suggestions = [w for w in self.word_list if w.startswith(last_word) and w != last_word][:4]
        for idx, suggestion in enumerate(suggestions):
            btn = Button(self.suggestion_frame, text=suggestion, width=10,
                         command=lambda s=suggestion: self.apply_suggestion(s))
            btn.grid(row=0, column=idx, padx=2)
            self.suggestion_buttons.append(btn)

    def apply_suggestion(self, suggestion):
        # Replace the last word with the selected suggestion
        sentence = self.sentence.get()
        words = sentence.split()
        if not words:
            return
        words[-1] = suggestion
        self.sentence.set(' '.join(words))
        self.update_suggestions()

    def add_space(self):
        # Add a space to the sentence
        self.sentence.set(self.sentence.get() + ' ')
        self.update_suggestions()

    def on_close(self):
        # Clean up resources and close the window
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.mainloop() 
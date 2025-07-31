#!/usr/bin/env python3
"""
Python 3.13 Compatible Gesture Detection System
A simplified but powerful gesture detection system that works with Python 3.13+
Uses TensorFlow and OpenCV for custom gesture training and detection.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import json
import time
from pathlib import Path
from PIL import Image, ImageTk
import threading
from datetime import datetime
import pickle

# Check TensorFlow availability
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} available")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available")

# Check scikit-learn
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class SimpleTensorFlowTrainer:
    """TensorFlow-based gesture trainer for Python 3.13."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.gesture_classes = {}
        self.training_data = []
        self.is_trained = False
        
    def create_cnn_model(self, num_classes, input_shape=(224, 224, 3)):
        """Create a CNN model for gesture classification."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_hand_features(self, frame):
        """Extract hand region and features from frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply skin color detection (simple method)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (assumed to be hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract hand region with padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            hand_region = frame[y1:y2, x1:x2]
            
            if hand_region.size > 0:
                # Resize to standard size
                hand_region = cv2.resize(hand_region, (224, 224))
                return hand_region, (x, y, w, h)
        
        # Return full frame if no hand detected
        return cv2.resize(frame, (224, 224)), None
    
    def collect_training_data(self, frame, gesture_label):
        """Collect training data from frame."""
        hand_region, bbox = self.extract_hand_features(frame)
        
        # Normalize pixel values
        hand_region = hand_region.astype(np.float32) / 255.0
        
        # Store training sample
        if gesture_label not in self.gesture_classes:
            self.gesture_classes[gesture_label] = len(self.gesture_classes)
        
        self.training_data.append({
            'image': hand_region,
            'label': self.gesture_classes[gesture_label],
            'gesture': gesture_label,
            'bbox': bbox
        })
        
        return bbox
    
    def train_model(self, epochs=50, validation_split=0.2):
        """Train the gesture recognition model."""
        if not self.training_data:
            raise ValueError("No training data available")
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for training")
        
        # Prepare data
        X = np.array([item['image'] for item in self.training_data])
        y = np.array([item['label'] for item in self.training_data])
        
        # Create model
        num_classes = len(self.gesture_classes)
        self.model = self.create_cnn_model(num_classes)
        
        # Add data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        # Train model
        history = self.model.fit(
            datagen.flow(X, y, batch_size=32),
            epochs=epochs,
            validation_split=validation_split,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict_gesture(self, frame):
        """Predict gesture from frame."""
        if not self.is_trained or self.model is None:
            return None, 0.0, None
        
        hand_region, bbox = self.extract_hand_features(frame)
        
        # Normalize
        hand_region = hand_region.astype(np.float32) / 255.0
        hand_region = np.expand_dims(hand_region, axis=0)
        
        # Predict
        predictions = self.model.predict(hand_region, verbose=0)
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        # Get gesture name
        gesture_name = None
        for gesture, class_id in self.gesture_classes.items():
            if class_id == predicted_class:
                gesture_name = gesture
                break
        
        return gesture_name, confidence, bbox
    
    def save_model(self, filepath):
        """Save trained model."""
        if self.model and self.is_trained:
            # Save TensorFlow model
            model_path = filepath.replace('.pkl', '_model')
            self.model.save(model_path)
            
            # Save metadata
            metadata = {
                'gesture_classes': self.gesture_classes,
                'model_path': model_path,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(metadata, f)
            
            return True
        return False
    
    def load_model(self, filepath):
        """Load trained model."""
        try:
            # Load metadata
            with open(filepath, 'rb') as f:
                metadata = pickle.load(f)
            
            self.gesture_classes = metadata['gesture_classes']
            self.is_trained = metadata['is_trained']
            
            # Load TensorFlow model
            if TF_AVAILABLE:
                self.model = tf.keras.models.load_model(metadata['model_path'])
                return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return False

class Python313GestureApp:
    """Python 3.13 compatible gesture detection application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Python 3.13 Gesture Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        
        # Initialize trainer
        self.trainer = SimpleTensorFlowTrainer()
        
        # Camera setup
        self.cap = None
        self.setup_camera()
        
        # Application state
        self.is_running = False
        self.is_training_mode = False
        self.is_recording = False
        self.current_gesture_label = "thumbs_up"
        self.frame_count = 0
        self.detection_results = []
        
        # Setup UI
        self.setup_ui()
        
        # Show compatibility message
        self.show_compatibility_message()
    
    def setup_camera(self):
        """Setup camera."""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception as e:
            print(f"Camera setup error: {e}")
    
    def show_compatibility_message(self):
        """Show Python 3.13 compatibility message."""
        message = (
            "🤖 Python 3.13 Compatible System\n\n"
            "✅ This version works with Python 3.13+\n"
            "✅ Uses TensorFlow for neural network training\n"
            "✅ Custom gesture training capabilities\n\n"
            "⚠️  For full DeepLabCut features, install Python 3.8-3.11\n\n"
            "Ready to train your custom gestures!"
        )
        messagebox.showinfo("System Info", message)
    
    def setup_ui(self):
        """Setup user interface."""
        # Configure grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Header
        self.setup_header()
        
        # Left panel
        self.setup_left_panel()
        
        # Center panel
        self.setup_center_panel()
        
        # Right panel
        self.setup_right_panel()
        
        # Status bar
        self.setup_status_bar()
    
    def setup_header(self):
        """Setup header."""
        header_frame = tk.Frame(self.root, bg='#2d2d2d', height=60)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=10, pady=5)
        header_frame.grid_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🤖 Python 3.13 Gesture Detection & Training",
            font=('Arial', 18, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        title_label.pack(side='left', padx=20, pady=15)
        
        # Mode toggle
        mode_frame = tk.Frame(header_frame, bg='#2d2d2d')
        mode_frame.pack(side='right', padx=20, pady=15)
        
        self.mode_var = tk.StringVar(value="training")
        
        training_radio = tk.Radiobutton(
            mode_frame,
            text="🎓 Training Mode",
            variable=self.mode_var,
            value="training",
            font=('Arial', 12),
            bg='#2d2d2d',
            fg='#ffffff',
            selectcolor='#4a4a4a',
            command=self.switch_mode
        )
        training_radio.pack(side='left', padx=10)
        
        detection_radio = tk.Radiobutton(
            mode_frame,
            text="🔍 Detection Mode",
            variable=self.mode_var,
            value="detection",
            font=('Arial', 12),
            bg='#2d2d2d',
            fg='#ffffff',
            selectcolor='#4a4a4a',
            command=self.switch_mode
        )
        detection_radio.pack(side='left', padx=10)
    
    def setup_left_panel(self):
        """Setup left control panel."""
        left_frame = tk.Frame(self.root, bg='#2a2a2a', width=300)
        left_frame.grid(row=1, column=0, sticky='ns', padx=(10, 5), pady=5)
        left_frame.grid_propagate(False)
        
        # Training controls
        training_frame = tk.LabelFrame(
            left_frame,
            text="🎓 Training Controls",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff'
        )
        training_frame.pack(fill='x', padx=10, pady=10)
        
        # Gesture label
        tk.Label(
            training_frame,
            text="Current Gesture:",
            font=('Arial', 10),
            bg='#2a2a2a',
            fg='#ffffff'
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        self.gesture_var = tk.StringVar(value="thumbs_up")
        gesture_combo = ttk.Combobox(
            training_frame,
            textvariable=self.gesture_var,
            values=[
                "thumbs_up", "peace", "ok", "fist", "open_palm",
                "pointing", "one", "two", "three", "four", "five",
                "smile", "frown", "wink", "custom"
            ],
            font=('Arial', 10)
        )
        gesture_combo.pack(fill='x', padx=10, pady=(0, 10))
        
        # Recording button
        self.record_button = tk.Button(
            training_frame,
            text="🔴 Start Recording",
            font=('Arial', 12, 'bold'),
            bg='#7c4a4a',
            fg='white',
            command=self.toggle_recording
        )
        self.record_button.pack(fill='x', padx=10, pady=(0, 10))
        
        # Training progress
        self.sample_count_label = tk.Label(
            training_frame,
            text="Samples collected: 0",
            font=('Arial', 10),
            bg='#2a2a2a',
            fg='#ffffff'
        )
        self.sample_count_label.pack(pady=(0, 10))
        
        # Train button
        self.train_button = tk.Button(
            training_frame,
            text="🏋️ Train Model",
            font=('Arial', 12, 'bold'),
            bg='#4a4a7c',
            fg='white',
            command=self.start_training,
            state='disabled'
        )
        self.train_button.pack(fill='x', padx=10, pady=(0, 15))
        
        # Model management
        model_frame = tk.LabelFrame(
            left_frame,
            text="🤖 Model Management",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff'
        )
        model_frame.pack(fill='x', padx=10, pady=10)
        
        save_button = tk.Button(
            model_frame,
            text="💾 Save Model",
            font=('Arial', 11, 'bold'),
            bg='#7c7c4a',
            fg='white',
            command=self.save_model
        )
        save_button.pack(fill='x', padx=10, pady=10)
        
        load_button = tk.Button(
            model_frame,
            text="📂 Load Model",
            font=('Arial', 11, 'bold'),
            bg='#7c7c4a',
            fg='white',
            command=self.load_model
        )
        load_button.pack(fill='x', padx=10, pady=(0, 15))
    
    def setup_center_panel(self):
        """Setup center camera panel."""
        center_frame = tk.Frame(self.root, bg='#1a1a1a')
        center_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        center_frame.grid_rowconfigure(1, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)
        
        # Camera controls
        controls_frame = tk.Frame(center_frame, bg='#2a2a2a')
        controls_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        
        self.start_button = tk.Button(
            controls_frame,
            text="🎥 Start Camera",
            font=('Arial', 12, 'bold'),
            bg='#4a7c59',
            fg='white',
            command=self.start_camera
        )
        self.start_button.pack(side='left', padx=10, pady=10)
        
        self.stop_button = tk.Button(
            controls_frame,
            text="⏹️ Stop Camera",
            font=('Arial', 12, 'bold'),
            bg='#7c4a4a',
            fg='white',
            command=self.stop_camera,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=10, pady=10)
        
        # Camera display
        camera_frame = tk.Frame(center_frame, bg='#000000', relief='sunken', bd=3)
        camera_frame.grid(row=1, column=0, sticky='nsew')
        camera_frame.grid_rowconfigure(0, weight=1)
        camera_frame.grid_columnconfigure(0, weight=1)
        
        self.camera_label = tk.Label(
            camera_frame,
            text="📷 Camera Feed\nClick 'Start Camera' to begin",
            font=('Arial', 16),
            bg='#000000',
            fg='#ffffff'
        )
        self.camera_label.grid(row=0, column=0, sticky='nsew')
    
    def setup_right_panel(self):
        """Setup right results panel."""
        right_frame = tk.Frame(self.root, bg='#2a2a2a', width=300)
        right_frame.grid(row=1, column=2, sticky='ns', padx=(5, 10), pady=5)
        right_frame.grid_propagate(False)
        
        # Detection results
        results_frame = tk.LabelFrame(
            right_frame,
            text="🎯 Detection Results",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff'
        )
        results_frame.pack(fill='x', padx=10, pady=10)
        
        self.gesture_result_label = tk.Label(
            results_frame,
            text="No gesture detected",
            font=('Arial', 16, 'bold'),
            bg='#2a2a2a',
            fg='#00ff00'
        )
        self.gesture_result_label.pack(pady=15)
        
        self.confidence_result_label = tk.Label(
            results_frame,
            text="Confidence: 0%",
            font=('Arial', 12),
            bg='#2a2a2a',
            fg='#ffffff'
        )
        self.confidence_result_label.pack(pady=(0, 15))
        
        # Training progress
        progress_frame = tk.LabelFrame(
            right_frame,
            text="📈 Training Progress",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff'
        )
        progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.progress_label = tk.Label(
            progress_frame,
            text="No training in progress",
            font=('Arial', 10),
            bg='#2a2a2a',
            fg='#ffffff'
        )
        self.progress_label.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate'
        )
        self.progress_bar.pack(fill='x', padx=10, pady=(0, 15))
        
        # System info
        info_frame = tk.LabelFrame(
            right_frame,
            text="🔧 System Info",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff'
        )
        info_frame.pack(fill='x', padx=10, pady=10)
        
        info_text = f"Python: {'.'.join(map(str, [3, 13]))}\n"
        info_text += f"TensorFlow: {'✅' if TF_AVAILABLE else '❌'}\n"
        info_text += f"OpenCV: ✅\n"
        info_text += f"Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}"
        
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=('Arial', 9),
            bg='#2a2a2a',
            fg='#ffffff',
            justify='left'
        )
        info_label.pack(pady=10)
    
    def setup_status_bar(self):
        """Setup status bar."""
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=30)
        status_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=10, pady=5)
        status_frame.grid_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="🤖 Python 3.13 System Ready | Training Mode",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#ffffff',
            anchor='w'
        )
        self.status_label.pack(fill='x', padx=10, pady=5)
    
    def switch_mode(self):
        """Switch between training and detection modes."""
        mode = self.mode_var.get()
        self.is_training_mode = (mode == "training")
        
        if self.is_training_mode:
            self.status_label.config(text="🎓 Training Mode | Collect gesture data")
        else:
            self.status_label.config(text="🔍 Detection Mode | Recognize trained gestures")
    
    def toggle_recording(self):
        """Toggle recording for training data."""
        self.is_recording = not self.is_recording
        
        if self.is_recording:
            self.current_gesture_label = self.gesture_var.get()
            self.record_button.config(text="⏸️ Stop Recording", bg='#4a7c4a')
            self.status_label.config(text=f"🔴 Recording: {self.current_gesture_label}")
        else:
            self.record_button.config(text="🔴 Start Recording", bg='#7c4a4a')
            self.status_label.config(text="⏹️ Recording stopped")
    
    def start_camera(self):
        """Start camera feed."""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return
        
        self.is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
    
    def stop_camera(self):
        """Stop camera feed."""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.camera_label.config(
            text="📷 Camera Feed\nClick 'Start Camera' to begin",
            image=''
        )
    
    def video_loop(self):
        """Main video processing loop."""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                if self.is_training_mode and self.is_recording:
                    # Collect training data
                    bbox = self.trainer.collect_training_data(frame, self.current_gesture_label)
                    
                    # Draw bounding box if detected
                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Recording: {self.current_gesture_label}", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Update sample count
                    sample_count = len([d for d in self.trainer.training_data 
                                      if d['gesture'] == self.current_gesture_label])
                    self.root.after(0, lambda: self.sample_count_label.config(
                        text=f"Samples collected: {sample_count}"
                    ))
                    
                    # Enable training button if enough samples
                    if len(self.trainer.training_data) >= 20:
                        self.root.after(0, lambda: self.train_button.config(state='normal'))
                
                elif not self.is_training_mode and self.trainer.is_trained:
                    # Perform detection
                    gesture, confidence, bbox = self.trainer.predict_gesture(frame)
                    
                    if gesture and confidence > 0.5:
                        # Draw bounding box
                        if bbox:
                            x, y, w, h = bbox
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.putText(frame, f"{gesture}: {confidence:.2f}", 
                                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Update UI
                        self.root.after(0, lambda g=gesture, c=confidence: self.update_detection_results(g, c))
                    else:
                        self.root.after(0, lambda: self.update_detection_results("No gesture", 0))
                
                # Update camera display
                self.update_camera_display(frame)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Video loop error: {e}")
                break
    
    def update_camera_display(self, frame):
        """Update camera display."""
        try:
            display_frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.root.after(0, lambda: self._update_camera_photo(photo))
        except Exception as e:
            print(f"Camera display error: {e}")
    
    def _update_camera_photo(self, photo):
        """Update camera photo in main thread."""
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo
    
    def update_detection_results(self, gesture, confidence):
        """Update detection results display."""
        self.gesture_result_label.config(text=gesture.replace('_', ' ').title())
        self.confidence_result_label.config(text=f"Confidence: {confidence:.1%}")
        
        # Set color based on confidence
        if confidence > 0.8:
            color = '#00ff00'
        elif confidence > 0.5:
            color = '#ffff00'
        else:
            color = '#ff7f7f'
        
        self.gesture_result_label.config(fg=color)
    
    def start_training(self):
        """Start model training."""
        if not TF_AVAILABLE:
            messagebox.showerror("Error", "TensorFlow not available for training")
            return
        
        if len(self.trainer.training_data) < 20:
            messagebox.showerror("Error", "Need at least 20 training samples")
            return
        
        self.progress_label.config(text="Training model...")
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()
        self.train_button.config(state='disabled')
        
        def training_thread():
            try:
                history = self.trainer.train_model(epochs=30)
                self.root.after(0, lambda: self.training_completed(True))
            except Exception as e:
                self.root.after(0, lambda: self.training_completed(False, str(e)))
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
    
    def training_completed(self, success, error_msg=None):
        """Handle training completion."""
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        self.train_button.config(state='normal')
        
        if success:
            self.progress_label.config(text="Training completed successfully!")
            messagebox.showinfo("Success", "Model training completed!\nSwitch to Detection Mode to test.")
        else:
            self.progress_label.config(text="Training failed")
            messagebox.showerror("Error", f"Training failed: {error_msg}")
    
    def save_model(self):
        """Save trained model."""
        if not self.trainer.is_trained:
            messagebox.showwarning("Warning", "No trained model to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")]
        )
        
        if filepath:
            if self.trainer.save_model(filepath):
                messagebox.showinfo("Success", "Model saved successfully!")
            else:
                messagebox.showerror("Error", "Failed to save model")
    
    def load_model(self):
        """Load trained model."""
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl")]
        )
        
        if filepath:
            if self.trainer.load_model(filepath):
                messagebox.showinfo("Success", "Model loaded successfully!")
                self.progress_label.config(text="Model loaded and ready")
            else:
                messagebox.showerror("Error", "Failed to load model")
    
    def run(self):
        """Run the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

def main():
    """Main function."""
    print("🤖 Python 3.13 Gesture Detection System")
    print("=" * 50)
    
    # Check dependencies
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available. Some features will be limited.")
        print("Install with: pip install tensorflow")
    
    try:
        app = Python313GestureApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application error: {e}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
DeepLabCut-based Gesture Detection System
Advanced gesture recognition with custom training capabilities for facial and hand gestures.
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, font, messagebox, filedialog
import threading
import time
import json
import pickle
from pathlib import Path
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from datetime import datetime

# DeepLabCut imports with fallback
try:
    import deeplabcut as dlc
    DLC_AVAILABLE = True
except ImportError:
    DLC_AVAILABLE = False
    print("DeepLabCut not available. Install with: pip install deeplabcut")

# Additional ML libraries
try:
    import tensorflow as tf
    import pandas as pd
    import sklearn
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML libraries not fully available. Install: pip install tensorflow pandas scikit-learn")

class DeepLabCutGestureTrainer:
    """Handles model training and data collection for custom gestures."""
    
    def __init__(self, project_path="./dlc_projects"):
        self.project_path = Path(project_path)
        self.project_path.mkdir(exist_ok=True)
        self.current_project = None
        self.config_path = None
        
        # Predefined body parts for different gesture types
        self.hand_bodyparts = [
            'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]
        
        self.face_bodyparts = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_eyebrow_inner', 'left_eyebrow_outer',
            'right_eyebrow_inner', 'right_eyebrow_outer',
            'mouth_left', 'mouth_right', 'mouth_top', 'mouth_bottom',
            'chin', 'forehead'
        ]
        
        self.gesture_database = {}
        self.load_gesture_database()
    
    def create_project(self, project_name, gesture_type="hand", experimenter="user"):
        """Create a new DeepLabCut project for gesture training."""
        if not DLC_AVAILABLE:
            raise ImportError("DeepLabCut not available")
        
        try:
            # Select appropriate body parts
            if gesture_type == "hand":
                bodyparts = self.hand_bodyparts
            elif gesture_type == "face":
                bodyparts = self.face_bodyparts
            else:  # combined
                bodyparts = self.hand_bodyparts + self.face_bodyparts
            
            # Create project
            config_path = dlc.create_new_project(
                project_name,
                experimenter,
                [],  # We'll add videos later
                working_directory=str(self.project_path),
                copy_videos=False,
                multianimal=False
            )
            
            # Update config with our body parts
            cfg = dlc.auxiliaryfunctions.read_config(config_path)
            cfg['bodyparts'] = bodyparts
            cfg['skeleton'] = self._generate_skeleton(bodyparts, gesture_type)
            cfg['default_net_type'] = 'resnet_50'
            cfg['default_augmenter'] = 'imgaug'
            
            # Save updated config
            dlc.auxiliaryfunctions.write_config(config_path, cfg)
            
            self.current_project = project_name
            self.config_path = config_path
            
            return config_path
            
        except Exception as e:
            raise Exception(f"Failed to create project: {e}")
    
    def _generate_skeleton(self, bodyparts, gesture_type):
        """Generate skeleton connections for visualization."""
        skeleton = []
        
        if gesture_type == "hand" or gesture_type == "combined":
            # Hand skeleton connections
            hand_connections = [
                ['wrist', 'thumb_cmc'], ['thumb_cmc', 'thumb_mcp'],
                ['thumb_mcp', 'thumb_ip'], ['thumb_ip', 'thumb_tip'],
                ['wrist', 'index_mcp'], ['index_mcp', 'index_pip'],
                ['index_pip', 'index_dip'], ['index_dip', 'index_tip'],
                ['wrist', 'middle_mcp'], ['middle_mcp', 'middle_pip'],
                ['middle_pip', 'middle_dip'], ['middle_dip', 'middle_tip'],
                ['wrist', 'ring_mcp'], ['ring_mcp', 'ring_pip'],
                ['ring_pip', 'ring_dip'], ['ring_dip', 'ring_tip'],
                ['wrist', 'pinky_mcp'], ['pinky_mcp', 'pinky_pip'],
                ['pinky_pip', 'pinky_dip'], ['pinky_dip', 'pinky_tip']
            ]
            skeleton.extend(hand_connections)
        
        if gesture_type == "face" or gesture_type == "combined":
            # Face skeleton connections
            face_connections = [
                ['left_eye', 'nose'], ['right_eye', 'nose'],
                ['left_ear', 'left_eye'], ['right_ear', 'right_eye'],
                ['left_eyebrow_inner', 'left_eyebrow_outer'],
                ['right_eyebrow_inner', 'right_eyebrow_outer'],
                ['mouth_left', 'mouth_right'], ['mouth_top', 'mouth_bottom'],
                ['nose', 'mouth_top'], ['chin', 'mouth_bottom']
            ]
            skeleton.extend(face_connections)
        
        return skeleton
    
    def collect_training_data(self, video_path, gesture_labels, num_frames=50):
        """Collect and label training data from video."""
        if not self.config_path:
            raise ValueError("No project created. Create a project first.")
        
        try:
            # Add video to project
            dlc.add_new_videos(self.config_path, [video_path])
            
            # Extract frames for labeling
            dlc.extract_frames(
                self.config_path,
                mode='automatic',
                algo='kmeans',
                userfeedback=False,
                crop=False,
                checkcropping=False
            )
            
            # Store gesture labels for this video
            video_name = Path(video_path).stem
            self.gesture_database[video_name] = {
                'labels': gesture_labels,
                'frames': num_frames,
                'timestamp': datetime.now().isoformat()
            }
            self.save_gesture_database()
            
            return True
            
        except Exception as e:
            print(f"Error collecting training data: {e}")
            return False
    
    def train_model(self, max_iters=100000, save_iters=10000):
        """Train the DeepLabCut model."""
        if not self.config_path:
            raise ValueError("No project created")
        
        try:
            # Create training dataset
            dlc.create_training_dataset(
                self.config_path,
                augmenter_type='imgaug'
            )
            
            # Train network
            dlc.train_network(
                self.config_path,
                shuffle=1,
                trainingsetindex=0,
                gputouse=None,  # Auto-detect GPU
                max_snapshots_to_keep=5,
                autotune=False,
                displayiters=1000,
                saveiters=save_iters,
                maxiters=max_iters
            )
            
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        if not self.config_path:
            raise ValueError("No project created")
        
        try:
            # Evaluate network
            dlc.evaluate_network(
                self.config_path,
                Shuffles=[1],
                plotting=True
            )
            
            return True
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return False
    
    def save_gesture_database(self):
        """Save gesture database to disk."""
        db_path = self.project_path / "gesture_database.json"
        with open(db_path, 'w') as f:
            json.dump(self.gesture_database, f, indent=2)
    
    def load_gesture_database(self):
        """Load gesture database from disk."""
        db_path = self.project_path / "gesture_database.json"
        if db_path.exists():
            with open(db_path, 'r') as f:
                self.gesture_database = json.load(f)

class CustomGestureRecognizer:
    """Advanced gesture recognition using DeepLabCut models."""
    
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.model_loaded = False
        self.gesture_classifiers = {}
        self.gesture_templates = {}
        self.confidence_threshold = 0.8
        
        if config_path and DLC_AVAILABLE:
            self.load_model()
    
    def load_model(self):
        """Load trained DeepLabCut model."""
        try:
            if self.config_path and os.path.exists(self.config_path):
                # Create pose estimation object
                self.dlc_live = dlc.live.DLCLive(
                    self.config_path,
                    processor=dlc.live.Processor(),
                    display=False
                )
                self.model_loaded = True
                print("DeepLabCut model loaded successfully")
            else:
                print("No valid model path provided")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def predict_pose(self, frame):
        """Predict pose keypoints from frame."""
        if not self.model_loaded:
            return None
        
        try:
            # Get pose prediction
            pose = self.dlc_live.get_pose(frame)
            return pose
        except Exception as e:
            print(f"Pose prediction error: {e}")
            return None
    
    def recognize_gesture(self, pose_data):
        """Recognize gesture from pose keypoints."""
        if pose_data is None:
            return None
        
        try:
            # Extract features from pose data
            features = self._extract_pose_features(pose_data)
            
            # Classify gesture using trained classifiers
            for gesture_name, classifier in self.gesture_classifiers.items():
                confidence = classifier.predict_proba([features])[0].max()
                if confidence > self.confidence_threshold:
                    return {
                        'gesture': gesture_name,
                        'confidence': confidence,
                        'keypoints': pose_data
                    }
            
            return None
            
        except Exception as e:
            print(f"Gesture recognition error: {e}")
            return None
    
    def _extract_pose_features(self, pose_data):
        """Extract meaningful features from pose keypoints."""
        features = []
        
        if len(pose_data) < 2:
            return np.zeros(100)  # Return zero vector for invalid data
        
        # Calculate distances between key points
        keypoints = pose_data.reshape(-1, 2)
        
        # Pairwise distances
        for i in range(len(keypoints)):
            for j in range(i + 1, min(i + 5, len(keypoints))):  # Limit to nearby points
                dist = np.linalg.norm(keypoints[i] - keypoints[j])
                features.append(dist)
        
        # Angles between triplets of points
        for i in range(len(keypoints) - 2):
            if i + 2 < len(keypoints):
                v1 = keypoints[i + 1] - keypoints[i]
                v2 = keypoints[i + 2] - keypoints[i + 1]
                
                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                features.append(angle)
        
        # Pad or truncate to fixed size
        features = np.array(features)
        if len(features) > 100:
            features = features[:100]
        elif len(features) < 100:
            features = np.pad(features, (0, 100 - len(features)), 'constant')
        
        return features
    
    def train_gesture_classifier(self, gesture_name, training_data):
        """Train a classifier for a specific gesture."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Prepare training data
            X = []
            y = []
            
            for pose_data, label in training_data:
                features = self._extract_pose_features(pose_data)
                X.append(features)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train classifier
            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            if len(np.unique(y)) > 1:  # Need at least 2 classes
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                classifier.fit(X_train, y_train)
                accuracy = classifier.score(X_test, y_test)
                
                self.gesture_classifiers[gesture_name] = classifier
                print(f"Trained {gesture_name} classifier with {accuracy:.2f} accuracy")
                
                return True
            else:
                print(f"Need more diverse training data for {gesture_name}")
                return False
                
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def save_classifiers(self, save_path):
        """Save trained classifiers."""
        with open(save_path, 'wb') as f:
            pickle.dump(self.gesture_classifiers, f)
    
    def load_classifiers(self, load_path):
        """Load trained classifiers."""
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.gesture_classifiers = pickle.load(f)

class DeepLabCutGestureApp:
    """Main application with DeepLabCut integration and training capabilities."""
    
    def __init__(self):
        """Initialize the DeepLabCut gesture application."""
        self.root = tk.Tk()
        self.root.title("🧠 DeepLabCut Gesture Detection & Training")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # Initialize components
        self.trainer = DeepLabCutGestureTrainer() if DLC_AVAILABLE else None
        self.recognizer = CustomGestureRecognizer()
        
        # Camera setup
        self.cap = None
        self.setup_camera()
        
        # Application state
        self.is_running = False
        self.is_training_mode = False
        self.current_gesture_label = "unknown"
        self.training_data = []
        self.gesture_history = []
        self.recording_session = False
        
        # UI setup
        self.setup_ui()
        self.check_dependencies()
        
        # Auto-load any existing models
        self.auto_load_models()
    
    def setup_camera(self):
        """Setup camera with enhanced error handling."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                for i in range(1, 4):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"Camera setup error: {e}")
            self.cap = None
    
    def check_dependencies(self):
        """Check and display dependency status."""
        status = []
        
        if DLC_AVAILABLE:
            status.append("✅ DeepLabCut: Available")
        else:
            status.append("❌ DeepLabCut: Not installed")
            
        if ML_AVAILABLE:
            status.append("✅ ML Libraries: Available")
        else:
            status.append("❌ ML Libraries: Incomplete")
        
        try:
            import cv2
            status.append("✅ OpenCV: Available")
        except ImportError:
            status.append("❌ OpenCV: Not available")
        
        print("\n".join(status))
    
    def setup_ui(self):
        """Setup comprehensive UI with training capabilities."""
        # Configure main grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Setup different panels
        self.setup_header()
        self.setup_left_panel()
        self.setup_center_panel()
        self.setup_right_panel()
        self.setup_status_bar()
    
    def setup_header(self):
        """Setup header with mode selection."""
        header_frame = tk.Frame(self.root, bg='#2d2d2d', height=80)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=10, pady=5)
        header_frame.grid_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🧠 DeepLabCut Gesture Detection & Training System",
            font=('Arial', 20, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        title_label.pack(side='left', padx=20, pady=20)
        
        # Mode selection
        mode_frame = tk.Frame(header_frame, bg='#2d2d2d')
        mode_frame.pack(side='right', padx=20, pady=20)
        
        self.mode_var = tk.StringVar(value="detection")
        
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
    
    def setup_left_panel(self):
        """Setup left panel for training controls."""
        left_frame = tk.Frame(self.root, bg='#2a2a2a', width=300)
        left_frame.grid(row=1, column=0, sticky='ns', padx=(10, 5), pady=5)
        left_frame.grid_propagate(False)
        
        # Training controls
        training_frame = tk.LabelFrame(
            left_frame,
            text="🎓 Training Controls",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff',
            relief='raised',
            bd=2
        )
        training_frame.pack(fill='x', padx=10, pady=10)
        
        # Project creation
        tk.Label(
            training_frame,
            text="Project Name:",
            font=('Arial', 10),
            bg='#2a2a2a',
            fg='#ffffff'
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        self.project_name_var = tk.StringVar(value="gesture_project")
        project_entry = tk.Entry(
            training_frame,
            textvariable=self.project_name_var,
            font=('Arial', 10),
            bg='#3a3a3a',
            fg='#ffffff',
            insertbackground='#ffffff'
        )
        project_entry.pack(fill='x', padx=10, pady=(0, 10))
        
        # Gesture type selection
        tk.Label(
            training_frame,
            text="Gesture Type:",
            font=('Arial', 10),
            bg='#2a2a2a',
            fg='#ffffff'
        ).pack(anchor='w', padx=10, pady=(0, 5))
        
        self.gesture_type_var = tk.StringVar(value="hand")
        gesture_type_combo = ttk.Combobox(
            training_frame,
            textvariable=self.gesture_type_var,
            values=["hand", "face", "combined"],
            state="readonly",
            font=('Arial', 10)
        )
        gesture_type_combo.pack(fill='x', padx=10, pady=(0, 10))
        
        # Create project button
        create_project_btn = tk.Button(
            training_frame,
            text="🆕 Create Project",
            font=('Arial', 11, 'bold'),
            bg='#4a7c59',
            fg='white',
            command=self.create_project,
            relief='raised',
            bd=2
        )
        create_project_btn.pack(fill='x', padx=10, pady=(0, 15))
        
        # Data collection
        data_frame = tk.LabelFrame(
            left_frame,
            text="📊 Data Collection",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff',
            relief='raised',
            bd=2
        )
        data_frame.pack(fill='x', padx=10, pady=10)
        
        # Current gesture label
        tk.Label(
            data_frame,
            text="Current Gesture:",
            font=('Arial', 10),
            bg='#2a2a2a',
            fg='#ffffff'
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        self.gesture_label_var = tk.StringVar(value="thumbs_up")
        gesture_label_combo = ttk.Combobox(
            data_frame,
            textvariable=self.gesture_label_var,
            values=[
                "thumbs_up", "peace", "ok", "fist", "open_palm",
                "pointing", "rock_on", "call_me", "smile", "frown",
                "wink", "surprise", "custom"
            ],
            font=('Arial', 10)
        )
        gesture_label_combo.pack(fill='x', padx=10, pady=(0, 10))
        
        # Recording controls
        self.record_button = tk.Button(
            data_frame,
            text="🔴 Start Recording",
            font=('Arial', 11, 'bold'),
            bg='#7c4a4a',
            fg='white',
            command=self.toggle_recording,
            relief='raised',
            bd=2
        )
        self.record_button.pack(fill='x', padx=10, pady=(0, 10))
        
        # Training button
        self.train_button = tk.Button(
            data_frame,
            text="🏋️ Train Model",
            font=('Arial', 11, 'bold'),
            bg='#4a4a7c',
            fg='white',
            command=self.start_training,
            relief='raised',
            bd=2,
            state='disabled'
        )
        self.train_button.pack(fill='x', padx=10, pady=(0, 15))
        
        # Model management
        model_frame = tk.LabelFrame(
            left_frame,
            text="🤖 Model Management",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff',
            relief='raised',
            bd=2
        )
        model_frame.pack(fill='x', padx=10, pady=10)
        
        load_model_btn = tk.Button(
            model_frame,
            text="📂 Load Model",
            font=('Arial', 11, 'bold'),
            bg='#7c7c4a',
            fg='white',
            command=self.load_model,
            relief='raised',
            bd=2
        )
        load_model_btn.pack(fill='x', padx=10, pady=10)
        
        save_model_btn = tk.Button(
            model_frame,
            text="💾 Save Model",
            font=('Arial', 11, 'bold'),
            bg='#7c7c4a',
            fg='white',
            command=self.save_model,
            relief='raised',
            bd=2
        )
        save_model_btn.pack(fill='x', padx=10, pady=(0, 15))
    
    def setup_center_panel(self):
        """Setup center panel for camera feed."""
        center_frame = tk.Frame(self.root, bg='#1a1a1a')
        center_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        center_frame.grid_rowconfigure(1, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)
        
        # Camera controls
        camera_controls = tk.Frame(center_frame, bg='#2a2a2a')
        camera_controls.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        
        self.start_button = tk.Button(
            camera_controls,
            text="🎥 Start Camera",
            font=('Arial', 12, 'bold'),
            bg='#4a7c59',
            fg='white',
            command=self.start_detection,
            relief='raised',
            bd=2
        )
        self.start_button.pack(side='left', padx=10, pady=10)
        
        self.stop_button = tk.Button(
            camera_controls,
            text="⏹️ Stop Camera",
            font=('Arial', 12, 'bold'),
            bg='#7c4a4a',
            fg='white',
            command=self.stop_detection,
            relief='raised',
            bd=2,
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
            fg='#ffffff',
            justify='center'
        )
        self.camera_label.grid(row=0, column=0, sticky='nsew')
    
    def setup_right_panel(self):
        """Setup right panel for results and history."""
        right_frame = tk.Frame(self.root, bg='#2a2a2a', width=350)
        right_frame.grid(row=1, column=2, sticky='ns', padx=(5, 10), pady=5)
        right_frame.grid_propagate(False)
        
        # Detection results
        results_frame = tk.LabelFrame(
            right_frame,
            text="🎯 Detection Results",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff',
            relief='raised',
            bd=2
        )
        results_frame.pack(fill='x', padx=10, pady=10)
        
        self.current_gesture_label = tk.Label(
            results_frame,
            text="No gesture detected",
            font=('Arial', 16, 'bold'),
            bg='#2a2a2a',
            fg='#00ff00',
            justify='center'
        )
        self.current_gesture_label.pack(pady=15)
        
        self.confidence_label = tk.Label(
            results_frame,
            text="Confidence: 0%",
            font=('Arial', 12),
            bg='#2a2a2a',
            fg='#ffffff'
        )
        self.confidence_label.pack(pady=(0, 15))
        
        # Training progress
        self.progress_frame = tk.LabelFrame(
            right_frame,
            text="📈 Training Progress",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff',
            relief='raised',
            bd=2
        )
        self.progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="No training in progress",
            font=('Arial', 10),
            bg='#2a2a2a',
            fg='#ffffff',
            justify='center'
        )
        self.progress_label.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            mode='determinate',
            length=300
        )
        self.progress_bar.pack(pady=(0, 15))
        
        # Gesture history
        history_frame = tk.LabelFrame(
            right_frame,
            text="📋 Gesture History",
            font=('Arial', 14, 'bold'),
            bg='#2a2a2a',
            fg='#ffffff',
            relief='raised',
            bd=2
        )
        history_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # History listbox with scrollbar
        history_list_frame = tk.Frame(history_frame, bg='#2a2a2a')
        history_list_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.history_listbox = tk.Listbox(
            history_list_frame,
            font=('Arial', 9),
            bg='#3a3a3a',
            fg='#ffffff',
            selectbackground='#4a4a4a',
            selectforeground='#ffffff',
            height=10
        )
        self.history_listbox.pack(side='left', fill='both', expand=True)
        
        history_scrollbar = tk.Scrollbar(
            history_list_frame,
            orient='vertical',
            command=self.history_listbox.yview
        )
        history_scrollbar.pack(side='right', fill='y')
        self.history_listbox.config(yscrollcommand=history_scrollbar.set)
    
    def setup_status_bar(self):
        """Setup bottom status bar."""
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=30)
        status_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=10, pady=5)
        status_frame.grid_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="🔴 Ready | DLC: Not loaded | Mode: Detection",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#ffffff',
            anchor='w'
        )
        self.status_label.pack(fill='x', padx=10, pady=5)
    
    def switch_mode(self):
        """Switch between detection and training modes."""
        mode = self.mode_var.get()
        self.is_training_mode = (mode == "training")
        
        if self.is_training_mode:
            self.status_label.config(text="🎓 Training Mode | Ready to collect data")
        else:
            self.status_label.config(text="🔍 Detection Mode | Ready for gesture recognition")
    
    def create_project(self):
        """Create a new DeepLabCut project."""
        if not DLC_AVAILABLE:
            messagebox.showerror("Error", "DeepLabCut not available. Install with: pip install deeplabcut")
            return
        
        project_name = self.project_name_var.get()
        gesture_type = self.gesture_type_var.get()
        
        if not project_name:
            messagebox.showerror("Error", "Please enter a project name")
            return
        
        try:
            self.progress_label.config(text="Creating project...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            
            # Create project in separate thread
            def create_project_thread():
                try:
                    config_path = self.trainer.create_project(project_name, gesture_type)
                    self.root.after(0, lambda: self.project_created_callback(config_path))
                except Exception as e:
                    self.root.after(0, lambda: self.project_error_callback(str(e)))
            
            thread = threading.Thread(target=create_project_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create project: {e}")
            self.progress_bar.stop()
            self.progress_label.config(text="Project creation failed")
    
    def project_created_callback(self, config_path):
        """Callback when project is created successfully."""
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        self.progress_label.config(text="Project created successfully!")
        
        # Load the new project
        self.recognizer.config_path = config_path
        self.train_button.config(state='normal')
        
        messagebox.showinfo("Success", f"Project created successfully!\nConfig: {config_path}")
    
    def project_error_callback(self, error_msg):
        """Callback when project creation fails."""
        self.progress_bar.stop()
        self.progress_label.config(text="Project creation failed")
        messagebox.showerror("Error", f"Failed to create project: {error_msg}")
    
    def toggle_recording(self):
        """Toggle data recording for training."""
        self.recording_session = not self.recording_session
        
        if self.recording_session:
            self.record_button.config(text="⏸️ Stop Recording", bg='#4a7c4a')
            current_gesture = self.gesture_label_var.get()
            self.status_label.config(text=f"🔴 Recording: {current_gesture}")
        else:
            self.record_button.config(text="🔴 Start Recording", bg='#7c4a4a')
            self.status_label.config(text="⏹️ Recording stopped")
    
    def start_training(self):
        """Start model training."""
        if not self.trainer or not self.trainer.config_path:
            messagebox.showerror("Error", "No project created. Create a project first.")
            return
        
        if len(self.training_data) < 10:
            messagebox.showerror("Error", "Not enough training data. Collect more samples.")
            return
        
        try:
            self.progress_label.config(text="Training model...")
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start()
            self.train_button.config(state='disabled')
            
            # Train in separate thread
            def training_thread():
                try:
                    success = self.trainer.train_model(max_iters=50000)
                    self.root.after(0, lambda: self.training_completed_callback(success))
                except Exception as e:
                    self.root.after(0, lambda: self.training_error_callback(str(e)))
            
            thread = threading.Thread(target=training_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")
            self.progress_bar.stop()
            self.train_button.config(state='normal')
    
    def training_completed_callback(self, success):
        """Callback when training is completed."""
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        self.train_button.config(state='normal')
        
        if success:
            self.progress_label.config(text="Training completed successfully!")
            messagebox.showinfo("Success", "Model training completed successfully!")
            
            # Load the trained model
            if self.recognizer.config_path:
                self.recognizer.load_model()
        else:
            self.progress_label.config(text="Training failed")
            messagebox.showerror("Error", "Model training failed")
    
    def training_error_callback(self, error_msg):
        """Callback when training fails."""
        self.progress_bar.stop()
        self.progress_label.config(text="Training failed")
        self.train_button.config(state='normal')
        messagebox.showerror("Error", f"Training failed: {error_msg}")
    
    def load_model(self):
        """Load a trained model."""
        config_path = filedialog.askopenfilename(
            title="Select DeepLabCut config file",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        if config_path:
            self.recognizer.config_path = config_path
            self.recognizer.load_model()
            
            if self.recognizer.model_loaded:
                messagebox.showinfo("Success", "Model loaded successfully!")
                self.status_label.config(text="✅ Model loaded | Ready for detection")
            else:
                messagebox.showerror("Error", "Failed to load model")
    
    def save_model(self):
        """Save trained classifiers."""
        if not self.recognizer.gesture_classifiers:
            messagebox.showwarning("Warning", "No trained classifiers to save")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Save classifiers",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if save_path:
            self.recognizer.save_classifiers(save_path)
            messagebox.showinfo("Success", "Classifiers saved successfully!")
    
    def auto_load_models(self):
        """Automatically load any existing models."""
        try:
            # Look for existing projects
            project_path = Path("./dlc_projects")
            if project_path.exists():
                for project_dir in project_path.iterdir():
                    if project_dir.is_dir():
                        config_files = list(project_dir.glob("**/config.yaml"))
                        if config_files:
                            config_path = str(config_files[0])
                            self.recognizer.config_path = config_path
                            print(f"Found existing project: {config_path}")
                            break
        except Exception as e:
            print(f"Auto-load error: {e}")
    
    def start_detection(self):
        """Start camera and detection."""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return
        
        self.is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
    
    def stop_detection(self):
        """Stop camera and detection."""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        # Reset camera display
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
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                if self.is_training_mode and self.recording_session:
                    # Collect training data
                    self.collect_training_sample(frame)
                else:
                    # Perform detection
                    self.detect_gesture(frame)
                
                # Update camera display
                self.update_camera_display(frame)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Video loop error: {e}")
                break
    
    def collect_training_sample(self, frame):
        """Collect training sample from current frame."""
        try:
            current_gesture = self.gesture_label_var.get()
            
            # Get pose estimation (if model is loaded)
            if self.recognizer.model_loaded:
                pose_data = self.recognizer.predict_pose(frame)
                if pose_data is not None:
                    self.training_data.append((pose_data, current_gesture))
                    
                    # Update UI
                    self.root.after(0, lambda: self.update_training_progress())
        except Exception as e:
            print(f"Training sample collection error: {e}")
    
    def detect_gesture(self, frame):
        """Detect gesture from current frame."""
        try:
            if self.recognizer.model_loaded:
                # Get pose prediction
                pose_data = self.recognizer.predict_pose(frame)
                
                if pose_data is not None:
                    # Recognize gesture
                    result = self.recognizer.recognize_gesture(pose_data)
                    
                    if result:
                        gesture_name = result['gesture']
                        confidence = result['confidence']
                        
                        # Update UI
                        self.root.after(0, lambda: self.update_detection_results(gesture_name, confidence))
                        
                        # Add to history
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self.gesture_history.append({
                            'gesture': gesture_name,
                            'confidence': confidence,
                            'timestamp': timestamp
                        })
                        
                        # Update history display
                        self.root.after(0, self.update_history_display)
                    else:
                        self.root.after(0, lambda: self.update_detection_results("No gesture", 0))
        except Exception as e:
            print(f"Gesture detection error: {e}")
    
    def update_camera_display(self, frame):
        """Update camera display with video feed."""
        try:
            # Resize for display
            display_frame = cv2.resize(frame, (640, 480))
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update display
            self.root.after(0, lambda: self._update_camera_photo(photo))
        except Exception as e:
            print(f"Camera display error: {e}")
    
    def _update_camera_photo(self, photo):
        """Update camera photo in main thread."""
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo
    
    def update_detection_results(self, gesture_name, confidence):
        """Update detection results display."""
        self.current_gesture_label.config(text=gesture_name.replace('_', ' ').title())
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
        
        # Update status
        if confidence > 0.8:
            color = '#00ff00'  # Green for high confidence
        elif confidence > 0.5:
            color = '#ffff00'  # Yellow for medium confidence
        else:
            color = '#ff7f7f'  # Light red for low confidence
        
        self.current_gesture_label.config(fg=color)
    
    def update_training_progress(self):
        """Update training progress display."""
        num_samples = len(self.training_data)
        self.progress_label.config(text=f"Collected {num_samples} training samples")
        
        if num_samples >= 10:
            self.train_button.config(state='normal')
    
    def update_history_display(self):
        """Update gesture history display."""
        # Keep only recent history
        if len(self.gesture_history) > 50:
            self.gesture_history = self.gesture_history[-50:]
        
        # Update listbox
        self.history_listbox.delete(0, tk.END)
        for entry in reversed(self.gesture_history[-20:]):  # Show last 20
            text = f"{entry['timestamp']} - {entry['gesture']} ({entry['confidence']:.1%})"
            self.history_listbox.insert(0, text)
    
    def run(self):
        """Run the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Show initial message
        if not DLC_AVAILABLE:
            messagebox.showwarning(
                "DeepLabCut Not Available",
                "DeepLabCut is not installed. Some features will be limited.\n\n"
                "Install with: pip install deeplabcut\n\n"
                "The app will run in basic mode using OpenCV."
            )
        
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
    try:
        app = DeepLabCutGestureApp()
        app.run()
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Application error: {e}")

if __name__ == "__main__":
    main() 
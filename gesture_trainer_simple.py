#!/usr/bin/env python3
"""
Simplified Gesture Trainer
Easy-to-use utility for collecting gesture data and training custom models.
"""

import cv2
import numpy as np
import os
import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog

class SimpleGestureTrainer:
    """Simplified gesture training system."""
    
    def __init__(self):
        self.data_dir = Path("gesture_training_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.cap = None
        self.current_gesture = "unknown"
        self.is_recording = False
        self.frame_count = 0
        self.session_data = []
        
        self.setup_camera()
        
    def setup_camera(self):
        """Setup camera."""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print("Camera initialized successfully")
            else:
                print("Failed to initialize camera")
        except Exception as e:
            print(f"Camera setup error: {e}")
    
    def collect_gesture_data(self):
        """Interactive data collection."""
        print("\n🎓 Gesture Training Data Collection")
        print("=" * 50)
        print("Instructions:")
        print("- Press SPACE to start/stop recording")
        print("- Press 'g' to change gesture label")
        print("- Press 'q' to quit")
        print("- Press 's' to save session data")
        print("=" * 50)
        
        if not self.cap or not self.cap.isOpened():
            print("❌ Camera not available")
            return
        
        gesture_count = {}
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add recording indicator
            if self.is_recording:
                cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save frame data
                frame_data = {
                    'gesture': self.current_gesture,
                    'timestamp': time.time(),
                    'frame_number': self.frame_count
                }
                self.session_data.append(frame_data)
                
                # Save frame image
                gesture_dir = self.data_dir / self.current_gesture
                gesture_dir.mkdir(exist_ok=True)
                
                frame_filename = gesture_dir / f"frame_{self.frame_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                
                self.frame_count += 1
                
                # Update gesture count
                if self.current_gesture not in gesture_count:
                    gesture_count[self.current_gesture] = 0
                gesture_count[self.current_gesture] += 1
            
            # Add text overlay
            cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if self.is_recording:
                cv2.putText(frame, f"Recording: {gesture_count.get(self.current_gesture, 0)} frames", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Press SPACE to record", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show gesture counts
            y_pos = 180
            for gesture, count in gesture_count.items():
                cv2.putText(frame, f"{gesture}: {count}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 25
            
            cv2.imshow('Gesture Training - Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to toggle recording
                self.is_recording = not self.is_recording
                if self.is_recording:
                    print(f"🔴 Started recording gesture: {self.current_gesture}")
                else:
                    print(f"⏹️ Stopped recording")
            
            elif key == ord('g'):  # Change gesture
                self.change_gesture_label()
            
            elif key == ord('s'):  # Save session
                self.save_session_data()
                print("💾 Session data saved")
            
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        print(f"\n📊 Session Summary:")
        for gesture, count in gesture_count.items():
            print(f"  {gesture}: {count} frames")
    
    def change_gesture_label(self):
        """Change current gesture label."""
        print("\nCurrent gesture labels collected:")
        
        gesture_dirs = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        for i, gesture in enumerate(gesture_dirs, 1):
            print(f"  {i}. {gesture}")
        
        new_gesture = input("\nEnter new gesture name (or press Enter to keep current): ").strip()
        if new_gesture:
            self.current_gesture = new_gesture
            print(f"Changed to gesture: {self.current_gesture}")
    
    def save_session_data(self):
        """Save session metadata."""
        session_file = self.data_dir / f"session_{int(time.time())}.json"
        
        session_info = {
            'timestamp': time.time(),
            'total_frames': len(self.session_data),
            'gestures': list(set(item['gesture'] for item in self.session_data)),
            'data': self.session_data
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_info, f, indent=2)
    
    def analyze_collected_data(self):
        """Analyze collected training data."""
        print("\n📊 Training Data Analysis")
        print("=" * 40)
        
        if not self.data_dir.exists():
            print("No training data found")
            return
        
        total_samples = 0
        gesture_stats = {}
        
        for gesture_dir in self.data_dir.iterdir():
            if gesture_dir.is_dir():
                frames = list(gesture_dir.glob("*.jpg"))
                count = len(frames)
                gesture_stats[gesture_dir.name] = count
                total_samples += count
        
        print(f"Total samples: {total_samples}")
        print(f"Unique gestures: {len(gesture_stats)}")
        print("\nGesture breakdown:")
        
        for gesture, count in sorted(gesture_stats.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {gesture:<15}: {count:>5} frames ({percentage:>5.1f}%)")
        
        # Recommendations
        print("\n💡 Recommendations:")
        min_samples = 100
        max_samples = 1000
        
        for gesture, count in gesture_stats.items():
            if count < min_samples:
                needed = min_samples - count
                print(f"  📈 {gesture}: Collect {needed} more samples (minimum: {min_samples})")
            elif count > max_samples:
                excess = count - max_samples
                print(f"  📉 {gesture}: Consider reducing by {excess} samples (maximum: {max_samples})")
            else:
                print(f"  ✅ {gesture}: Good sample count")
    
    def create_training_dataset(self):
        """Create training dataset from collected data."""
        print("\n🔄 Creating training dataset...")
        
        if not self.data_dir.exists():
            print("No training data found")
            return None
        
        dataset = []
        labels = []
        gesture_to_id = {}
        
        # Create gesture ID mapping
        gesture_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        for i, gesture_dir in enumerate(gesture_dirs):
            gesture_to_id[gesture_dir.name] = i
        
        print(f"Found {len(gesture_to_id)} gesture classes:")
        for gesture, id in gesture_to_id.items():
            print(f"  {id}: {gesture}")
        
        # Load all images and labels
        for gesture_dir in gesture_dirs:
            gesture_name = gesture_dir.name
            gesture_id = gesture_to_id[gesture_name]
            
            image_files = list(gesture_dir.glob("*.jpg"))
            print(f"Loading {len(image_files)} samples for {gesture_name}...")
            
            for img_file in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        # Resize to standard size
                        img_resized = cv2.resize(img, (224, 224))
                        # Normalize
                        img_normalized = img_resized.astype(np.float32) / 255.0
                        
                        dataset.append(img_normalized)
                        labels.append(gesture_id)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
        
        if dataset:
            dataset = np.array(dataset)
            labels = np.array(labels)
            
            print(f"Dataset created: {dataset.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Save dataset
            dataset_file = self.data_dir / "training_dataset.npz"
            np.savez(dataset_file, 
                    images=dataset, 
                    labels=labels, 
                    gesture_to_id=gesture_to_id)
            
            print(f"Dataset saved to: {dataset_file}")
            return dataset_file
        else:
            print("No valid data found")
            return None
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        print("\n🚀 Complete Training Pipeline")
        print("=" * 50)
        
        # Step 1: Analyze current data
        self.analyze_collected_data()
        
        # Step 2: Ask if user wants to collect more data
        collect_more = input("\nDo you want to collect more training data? (y/n): ").lower() == 'y'
        
        if collect_more:
            self.collect_gesture_data()
        
        # Step 3: Create training dataset
        dataset_file = self.create_training_dataset()
        
        if dataset_file:
            print(f"\n✅ Training pipeline completed!")
            print(f"Dataset ready at: {dataset_file}")
            print("\nNext steps:")
            print("1. Use the dataset with your preferred ML framework")
            print("2. Train a CNN model for gesture classification")
            print("3. Integrate the trained model with the detection system")
        else:
            print("\n❌ Training pipeline failed - no valid dataset created")

def main():
    """Main function for interactive training."""
    trainer = SimpleGestureTrainer()
    
    print("\n🎓 Simple Gesture Trainer")
    print("=" * 30)
    print("Options:")
    print("1. Collect training data")
    print("2. Analyze collected data")
    print("3. Create training dataset")
    print("4. Run complete pipeline")
    print("5. Exit")
    
    while True:
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == '1':
            trainer.collect_gesture_data()
        elif choice == '2':
            trainer.analyze_collected_data()
        elif choice == '3':
            trainer.create_training_dataset()
        elif choice == '4':
            trainer.run_training_pipeline()
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please select 1-5.")
    
    if trainer.cap:
        trainer.cap.release()
    cv2.destroyAllWindows()
    print("Training session ended.")

if __name__ == "__main__":
    main() 
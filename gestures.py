"""
Gesture configuration and mapping for sign language detection.
This file contains the definitions for various hand gestures and their corresponding prompts.
"""

import math

class GestureRecognizer:
    def __init__(self):
        # Define gesture mappings
        self.gesture_prompts = {
            'thumbs_up': "Great job! 👍",
            'peace': "Peace and love! ✌️",
            'ok': "Everything is OK! 👌",
            'pointing_up': "Look up there! ☝️",
            'open_palm': "Hello! 👋",
            'fist': "Strong and determined! ✊",
            'call_me': "Call me! 🤙",
            'rock_on': "Rock on! 🤘",
            'one': "Number One! 1️⃣",
            'two': "Number Two! 2️⃣",
            'three': "Number Three! 3️⃣",
            'four': "Number Four! 4️⃣",
            'five': "Number Five! 5️⃣"
        }
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_finger_extended(self, landmarks, finger_tips, finger_pips, finger_mcps):
        """Check if a finger is extended based on landmark positions."""
        tip = landmarks[finger_tips]
        pip = landmarks[finger_pips]
        mcp = landmarks[finger_mcps]
        
        # Check if tip is above pip (extended)
        return tip.y < pip.y and tip.y < mcp.y
    
    def is_thumb_extended(self, landmarks):
        """Check if thumb is extended."""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        thumb_cmc = landmarks[1]
        
        # Thumb extension is different (horizontal movement)
        # Use both x-axis distance and relative positioning for better accuracy
        horizontal_extended = thumb_tip.x > thumb_ip.x and thumb_tip.x > thumb_mcp.x
        
        # Also check distance from palm center for better detection
        distance_from_palm = self.calculate_distance(
            (thumb_tip.x, thumb_tip.y), (thumb_cmc.x, thumb_cmc.y)
        )
        distance_threshold = 0.08  # Adjust based on hand size
        
        return horizontal_extended and distance_from_palm > distance_threshold
    
    def recognize_gesture(self, landmarks):
        """
        Recognize gesture based on hand landmarks.
        landmarks: List of hand landmarks from MediaPipe
        """
        if not landmarks:
            return None
        
        # Count extended fingers
        fingers_up = []
        
        # Thumb (special case)
        fingers_up.append(self.is_thumb_extended(landmarks))
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_pips = [6, 10, 14, 18]  # PIP joints
        finger_mcps = [5, 9, 13, 17]   # MCP joints
        
        for i in range(4):
            fingers_up.append(self.is_finger_extended(landmarks, finger_tips[i], finger_pips[i], finger_mcps[i]))
        
        # Count total extended fingers
        total_fingers = sum(fingers_up)
        
        # Recognize specific gestures
        if total_fingers == 1:
            if fingers_up[0]:  # Only thumb
                return 'thumbs_up'
            elif fingers_up[1]:  # Only index finger
                return 'pointing_up'
        
        elif total_fingers == 2:
            if fingers_up[1] and fingers_up[2]:  # Index and middle
                return 'peace'
            elif fingers_up[0] and fingers_up[4]:  # Thumb and pinky
                return 'call_me'
        
        elif total_fingers == 3:
            if fingers_up[0] and fingers_up[1] and fingers_up[2]:  # Thumb, index, middle
                return 'ok'
            elif fingers_up[1] and fingers_up[2] and fingers_up[3]:  # Index, middle, ring
                return 'three'
        
        elif total_fingers == 4:
            if not fingers_up[0]:  # All except thumb
                return 'four'
        
        elif total_fingers == 5:
            return 'five'
        
        elif total_fingers == 0:
            return 'fist'
        
        # Check for rock on gesture (index and pinky up, thumb down)
        if fingers_up[1] and fingers_up[4] and not fingers_up[2] and not fingers_up[3]:
            return 'rock_on'
        
        # Check for number gestures
        if total_fingers == 1 and fingers_up[1]:
            return 'one'
        elif total_fingers == 2 and fingers_up[1] and fingers_up[2]:
            return 'two'
        
        # Default to open palm for multiple fingers
        if total_fingers >= 3:
            return 'open_palm'
        
        return None
    
    def get_prompt(self, gesture):
        """Get the prompt text for a recognized gesture."""
        return self.gesture_prompts.get(gesture, "Unknown gesture detected") 
# 🤟 Sign Language & Gesture Detection Application

A real-time gesture and sign language detection application that uses computer vision to recognize hand gestures from your camera and displays corresponding text prompts. Built with Python, OpenCV, MediaPipe, and Tkinter.

## ✨ Features

- **Real-time gesture detection** using MediaPipe hand tracking
- **Multiple gesture recognition** including ASL signs and common gestures
- **Live camera feed** display in the application
- **Interactive GUI** with modern design
- **Gesture history tracking** with analytics
- **Confidence scoring** for gesture stability
- **Export functionality** for gesture data
- **Performance monitoring** with FPS counter

## 🎯 Supported Gestures

| Gesture | Prompt | Description |
|---------|--------|-------------|
| 👍 | "Great job! 👍" | Thumbs up |
| ✌️ | "Peace and love! ✌️" | Peace sign |
| 👌 | "Everything is OK! 👌" | OK sign |
| ☝️ | "Look up there! ☝️" | Pointing up |
| 👋 | "Hello! 👋" | Open palm wave |
| ✊ | "Strong and determined! ✊" | Closed fist |
| 🤙 | "Call me! 🤙" | Call me gesture |
| 🤘 | "Rock on! 🤘" | Rock and roll sign |
| 1️⃣-5️⃣ | "Number X!" | Number counting (1-5) |

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- A working webcam/camera
- Windows/Mac/Linux operating system

### Step 1: Clone or Download

Download the project files to your local machine.

### Step 2: Install Dependencies

Open a terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

### Step 3: Verify Camera Access

Make sure your camera is working and not being used by other applications.

## 🎮 Usage

### Basic Version

Run the basic gesture detector:

```bash
python gesture_detector.py
```

### Enhanced Version (Recommended)

Run the enhanced version with camera display:

```bash
python enhanced_detector.py
```

### How to Use

1. **Start the Application**: Click the "🎥 Start Detection" button
2. **Position Yourself**: Sit about 2-3 feet from your camera
3. **Show Gestures**: Make clear hand gestures toward the camera
4. **View Prompts**: Watch as the application displays corresponding prompts
5. **Check History**: Click "📋 View Gesture History" to see your gesture log
6. **Stop Detection**: Click "⏹️ Stop Detection" when finished

## 📁 Project Structure

```
SignLanguagePrompting/
├── gesture_detector.py      # Basic version of the application
├── enhanced_detector.py     # Enhanced version with camera display
├── gestures.py             # Gesture recognition logic
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Gesture Sensitivity

You can adjust gesture detection sensitivity in `gestures.py`:

```python
# In gesture_detector.py or enhanced_detector.py
self.required_stability = 8  # Number of frames to confirm gesture
```

### Camera Settings

Modify camera resolution in the detector files:

```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Detection Confidence

Adjust MediaPipe detection confidence:

```python
self.hands = self.mp_hands.Hands(
    min_detection_confidence=0.7,  # Adjust this value (0.0 - 1.0)
    min_tracking_confidence=0.5    # Adjust this value (0.0 - 1.0)
)
```

## 🎨 Features Comparison

| Feature | Basic Version | Enhanced Version |
|---------|---------------|------------------|
| Gesture Detection | ✅ | ✅ |
| Text Prompts | ✅ | ✅ |
| Camera Feed Display | ❌ | ✅ |
| Real-time FPS Counter | ❌ | ✅ |
| Confidence Scoring | ❌ | ✅ |
| Enhanced History View | ❌ | ✅ |
| Data Export | ❌ | ✅ |
| Modern UI Layout | Basic | Advanced |

## 🛠️ Troubleshooting

### Camera Issues

- **Camera not found**: Make sure your camera is connected and not in use
- **Permission denied**: Grant camera permissions to Python/terminal
- **Poor detection**: Ensure good lighting and clear background

### Performance Issues

- **Low FPS**: Close other applications using the camera
- **High CPU usage**: Reduce camera resolution or detection confidence
- **Lag in detection**: Increase `required_stability` value

### Common Errors

1. **ImportError**: Install missing dependencies with `pip install -r requirements.txt`
2. **Camera error**: Try different camera indices (0, 1, 2) in the code
3. **Tkinter issues**: Install tkinter: `sudo apt-get install python3-tk` (Linux)

## 🔮 Future Enhancements

- [ ] Support for more ASL alphabet letters
- [ ] Custom gesture training capability
- [ ] Voice output for prompts
- [ ] Multi-language support
- [ ] Gesture sequence recognition
- [ ] Integration with accessibility tools
- [ ] Mobile app version
- [ ] Remote control functionality

## 🤝 Contributing

Feel free to contribute to this project by:

1. Adding new gesture recognition patterns
2. Improving the UI/UX design
3. Optimizing performance
4. Adding new features
5. Fixing bugs or issues

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking technology
- **OpenCV** for computer vision capabilities
- **Tkinter** for GUI framework
- **PIL/Pillow** for image processing

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify your camera is working with other applications
4. Try both the basic and enhanced versions

---

**Happy Gesture Detection! 🤟** 
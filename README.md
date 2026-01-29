
# Voice Guided Object Detection and Navigation System for Visually Impaired

This project is an AI-powered mobile application designed to assist visually impaired users with indoor navigation using real-time object detection, depth estimation, and voice-guided instructions. The system integrates computer vision and speech technologies to enable users to identify nearby objects and move toward them safely using a smartphone camera and audio feedback.


## Project Overview

Traditional navigation aids such as white canes and GPS-based systems provide limited support in indoor environments. Although modern applications can describe scenes, they often lack real-time spatial awareness and navigation assistance.

This project addresses that gap by combining object detection, monocular depth estimation, and voice interaction into a single Android application. Users can issue voice commands such as “find chair,” after which the system detects the object, estimates its distance, and provides continuous, step-wise audio guidance while avoiding obstacles.

## Key Features

* Voice-based object search
* Real-time object detection using YOLOv8
* Monocular depth estimation using MiDaS
* Distance-aware navigation logic
* Step-wise audio guidance
* Obstacle detection and avoidance
* Offline execution using TensorFlow Lite
* Android mobile deployment

## System Workflow

1. The camera captures real-time frames.
2. YOLOv8 detects objects in the scene.
3. MiDaS estimates relative depth for distance calculation.
4. Navigation logic determines safe movement.
5. Text-to-speech generates directional feedback.
6. The user follows audio instructions to reach the target object.

Interaction is performed using voice commands and a double-tap gesture.

### Software

* Python
* Kotlin
* PyTorch
* OpenCV
* TensorFlow Lite
* YOLOv8
* MiDaS v2.1
* Android SpeechRecognizer API
* Android Text-to-Speech

### Hardware

* Android smartphone (Android 8.0+, 4GB RAM recommended)
* Rear camera
* Microphone
* Speaker or earphones

---


## Setup and Usage

### Python Environment

```bash
pip install torch opencv-python ultralytics tensorflow
```

### Train YOLO Model

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=50
```

### Export to TensorFlow Lite

```bash
yolo export model=yolov8n.pt format=tflite
```

### Android Deployment

1. Open the Android project in Android Studio.
2. Place the exported `.tflite` models into the `assets` directory.
3. Build and run the application on a physical device.

## How the System Works

The application continuously analyzes camera frames to detect objects and estimate depth. When the user speaks a command, the system locates the requested object and calculates its relative distance. Based on this information, navigation rules generate safe movement instructions that are delivered using audio feedback.

Example outputs include:

* “Move forward two steps.”
* “Turn slightly left.”
* “Obstacle ahead. Stop.”

## Use Cases

* Indoor navigation assistance
* Object locating for visually impaired users
* Assistive technology research
* AI for social good applications


## Future Enhancements

* Outdoor navigation support
* Multi-language voice interaction
* Improved obstacle mapping
* Wearable device integration
* Cloud-assisted model updates

## License

This project is intended for academic and research purposes.




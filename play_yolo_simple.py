import numpy as np
import cv2
import math
import torch
from server.config import Config
from server.object_detector import ObjectDetector
from server.frame_processor import FrameProcessor

print("YOLO Model Path:", Config.YOLO_MODEL_PATH)
print("Class Names:", Config.CLASS_NAMES)

# Initialize the ObjectDetector
detector = ObjectDetector(Config.YOLO_MODEL_PATH, Config.CLASS_NAMES)

# Load data from .npz files
base_data = np.load('base.npz')
test_data = np.load('test.npz')

# Extract images, GPS, and compass
base_images = base_data['images']
test_images = test_data['images']
base_gps = base_data['gps']
test_gps = test_data['gps']
base_compass = base_data['compass']
test_compass = test_data['compass']

# Colors
colorBlue = (255, 0, 0)
colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

# Create a canvas (a black image) of 800x800 pixels, with 3 color channels (RGB), to represent the vehicle's position and orientation.
canvas = np.zeros((800, 800, 3))

# Process each pair of frames
for i in range(base_images.shape[0]):
    # Clear canvas
    canvas[:, :, :] = 0

    # Read next 'frame' from file
    # gps, compass = base_gps[i], base_compass[i]
    gps, compass = test_gps[i], test_compass[i]

    base_frame = base_images[i]
    test_frame = test_images[i]
    # change_frame = np.zeros_like(base_frame) # black image
    change_frame = test_frame.copy()  # clone the test image

    # Convert images from RGBA to RGB
    base_frame_rgb = cv2.cvtColor(base_frame, cv2.COLOR_RGBA2RGB)
    test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_RGBA2RGB)

    # Detect objects in base and test frames
    base_detections = detector.detect_objects(base_frame_rgb)
    test_detections = detector.detect_objects(test_frame_rgb)

    # Compare detections to identify changes
    changes = detector.compare_detections(base_detections, test_detections)

    # Visualize results
    detector.draw_detections(base_frame, base_detections, color=colorRed)
    detector.draw_detections(test_frame, test_detections, color=colorGreen)
    detector.draw_detections(change_frame, changes['appeared'], color=colorGreen, label_prefix='', isConfidence=False)
    detector.draw_detections(change_frame, changes['disappeared'], color=colorRed, label_prefix='Lost: ', isConfidence=False)
    detector.draw_detections(change_frame, changes['moved'], color=(0, 255, 255), label_prefix='Moved: ', isConfidence=False)

    # Draw the position and orientation of the vehicle on the canvas
    canvas = FrameProcessor.draw_compass_gps(canvas, gps, compass)

    # Show the images with the detections
    cv2.imshow('Base Objects', base_frame)
    cv2.imshow('Test Objects', test_frame)
    cv2.imshow('Changes Detected', change_frame)
    cv2.imshow('Canvas', canvas)

    k = cv2.waitKey(10)
    if k == 113 or k == 27:
        break

# Close all windows when finished
cv2.destroyAllWindows()
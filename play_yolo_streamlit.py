import numpy as np
import cv2
import streamlit as st
import time
import folium
from streamlit_folium import st_folium
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

# State of the current frame index and playback state
if 'frame_index' not in st.session_state:
    st.session_state.frame_index = 0

# Streamlit interface
st.title("Change Detection")

# Buttons to navigate through frames
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Previous"):
        st.session_state.frame_index = max(0, st.session_state.frame_index - 1)
with col4:
    if st.button("Next"):
        st.session_state.frame_index = min(base_images.shape[0] - 1, st.session_state.frame_index + 1)

# Get the current frame index
i = st.session_state.frame_index

# Create a canvas (a black image) of 800x800 pixels, with 3 color channels (RGB), to represent the vehicle's position and orientation.
canvas = np.zeros((800, 800, 3))

# Clear canvas
canvas[:, :, :] = 0

# Read next 'frame' from file
# gps, compass = base_gps[i], base_compass[i]
gps, compass = test_gps[i], test_compass[i]

# Get the current frames
base_frame = base_images[i]
test_frame = test_images[i]

# Get the current GPS coordinates
base_gps_coords = base_gps[i]
test_gps_coords = test_gps[i]

# Convert images from RGBA to RGB
base_frame_rgb = cv2.cvtColor(base_frame, cv2.COLOR_RGBA2RGB)
test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_RGBA2RGB)

# Detect objects in base and test frames
base_detections = detector.detect_objects(base_frame_rgb)
test_detections = detector.detect_objects(test_frame_rgb)

# Compare detections to identify changes
changes = detector.compare_detections(base_detections, test_detections)

# Clone the test image to display changes
change_frame = test_frame_rgb.copy()

# Visualize results
detector.draw_detections(base_frame_rgb, base_detections, color=colorRed)
detector.draw_detections(test_frame_rgb, test_detections, color=colorGreen)
detector.draw_detections(change_frame, changes['appeared'], color=colorGreen, label_prefix='', isConfidence=False)
detector.draw_detections(change_frame, changes['disappeared'], color=colorRed, label_prefix='Lost: ', isConfidence=False)
detector.draw_detections(change_frame, changes['moved'], color=(0, 255, 255), label_prefix='Moved: ', isConfidence=False)

# Draw the position and orientation of the vehicle on the canvas
canvas = FrameProcessor.draw_compass_gps(canvas, gps, compass)

# Convert the canvas to the appropriate depth for conversion
canvas = np.uint8(canvas)

# Convert frames to RGB for display in Streamlit
base_frame_rgb = cv2.cvtColor(base_frame_rgb, cv2.COLOR_BGR2RGB)
test_frame_rgb = cv2.cvtColor(test_frame_rgb, cv2.COLOR_BGR2RGB)
change_frame_rgb = cv2.cvtColor(change_frame, cv2.COLOR_BGR2RGB)
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# Create the map with Folium
map = folium.Map(location=[base_gps_coords[0], base_gps_coords[1]], zoom_start=5)
folium.Marker([base_gps_coords[0], base_gps_coords[1]], tooltip='Base GPS', icon=folium.Icon(color='red')).add_to(map)
folium.Marker([test_gps_coords[0], test_gps_coords[1]], tooltip='Test GPS', icon=folium.Icon(color='blue')).add_to(map)

# Display frames in Streamlit side by side
col1, col2 = st.columns(2)
with col1:
    st.image(base_frame_rgb, caption="Base Objects", use_column_width=True)
with col2:
    st.image(test_frame_rgb, caption="Test Objects", use_column_width=True)

st.image(change_frame_rgb, caption="Changes Detected", use_column_width=True)

col11, col22 = st.columns(2)
with col11:
    st_folium(map, width=350) # Display the map in Streamlit
with col22:
    st.image(canvas_rgb, caption="Compass/GPS", use_column_width=True)

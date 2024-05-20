from flask import Flask, render_template, Response, jsonify, request
import numpy as np
import cv2
from server.utils import *
from server.config import Config
from server.object_detector import ObjectDetector
from server.frame_processor import FrameProcessor

print("YOLO Model Path:", Config.YOLO_MODEL_PATH)
print("Class Names:", Config.CLASS_NAMES)

# Initialize Flask
app = Flask(__name__)

# Load data from .npz files
base_images, test_images, base_gps, test_gps, base_compass, test_compass = load_data('base.npz', 'test.npz')
canvas = np.zeros((800, 800, 3))

# Variable to keep track of the current frame index
current_frame = 0

# Initialize the ObjectDetector
detector = ObjectDetector(Config.YOLO_MODEL_PATH, Config.CLASS_NAMES)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/previous_frame')
def previous_frame():
    global current_frame
    current_frame = (current_frame - 1) % base_images.shape[0]
    return ('', 204)

@app.route('/next_frame')
def next_frame():
    global current_frame
    current_frame = (current_frame + 1) % base_images.shape[0]
    return ('', 204)

@app.route('/reset_frame')
def reset_frame():
    global current_frame
    current_frame = 0
    return ('', 204)

@app.route('/select_frame')
def select_frame():
    idFrame = request.args.get('frame_id')
    global current_frame
    current_frame = int(idFrame)
    return ('', 204)

@app.route('/gps_data')
def gps_data():
    base_point = {'lat': float(base_gps[current_frame][0]), 'lng': float(base_gps[current_frame][1]), 'type': 'base'}
    test_point = {'lat': float(test_gps[current_frame][0]), 'lng': float(test_gps[current_frame][1]), 'type': 'test'}
    return jsonify([base_point, test_point])

@app.route('/current_index')
def current_index():
    return str(current_frame)

@app.route('/size_frames')
def size_frames():
    return str(len(test_images))

def generate_frame(frame_type):
    if frame_type == 'base':
        frame = FrameProcessor.process_base_frame(base_images[current_frame], detector)
    elif frame_type == 'test':
        frame = FrameProcessor.process_test_frame(test_images[current_frame], detector)
    elif frame_type == 'change':
        frame = FrameProcessor.process_change_frame(base_images[current_frame], test_images[current_frame], detector)
    elif frame_type == 'compass':
        frame = FrameProcessor.process_compass_frame(test_gps[current_frame], test_compass[current_frame], canvas)

    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

@app.route('/base_frame_feed')
def base_frame_feed():
    return Response(generate_frame('base'), mimetype='image/jpeg')

@app.route('/test_frame_feed')
def test_frame_feed():
    return Response(generate_frame('test'), mimetype='image/jpeg')

@app.route('/change_frame_feed')
def change_frame_feed():
    return Response(generate_frame('change'), mimetype='image/jpeg')

@app.route('/compass_frame_feed')
def compass_frame_feed():
    return Response(generate_frame('compass'), mimetype='image/jpeg')

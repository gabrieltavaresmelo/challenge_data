import cv2
from ultralytics import YOLO
import time
import torch
import platform

class ObjectDetector:
    def __init__(self, model_path, class_names):
        system = platform.system()
        print(f"Running on {system}")
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.class_names = class_names

        time.sleep(1)
        self.model('server/static/temp.png') # to load YOLO model on init

    def detect_objects(self, frame):
        results = self.model(frame)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, cls, class_name, conf))
        return detections

    def compare_detections(self, base_detections, test_detections):
        changes = {
            'appeared': [],
            'disappeared': [],
            'moved': []
        }

        base_dict = {(d[0], d[1], d[2], d[3], d[4]): d for d in base_detections}
        test_dict = {(d[0], d[1], d[2], d[3], d[4]): d for d in test_detections}

        for key in base_dict:
            if key not in test_dict:
                changes['disappeared'].append(base_dict[key])

        for key in test_dict:
            if key not in base_dict:
                changes['appeared'].append(test_dict[key])

        for key in base_dict:
            if key in test_dict and base_dict[key] != test_dict[key]:
                changes['moved'].append(test_dict[key])

        return changes

    def draw_detections(self, frame, detections, color=(0, 255, 0), label_prefix='', isConfidence=True):
        for detection in detections:
            x1, y1, x2, y2, cls, class_name, conf = detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            if isConfidence:
                label = f"{label_prefix}{class_name} {conf:.2f}"
            else:
                label = f"{label_prefix}{class_name}"

            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


import cv2
import numpy as np

class FrameProcessor:
    @staticmethod
    def process_base_frame(base_frame, detector):
        base_frame_ori = base_frame
        base_frame = base_frame_ori.copy()
        base_frame_rgb = FrameProcessor.get_frame_rgb(base_frame)
        base_detections = detector.detect_objects(base_frame_rgb)
        detector.draw_detections(base_frame, base_detections, color=(0, 0, 255))
        return base_frame

    @staticmethod
    def process_test_frame(test_frame, detector):
        test_frame_ori = test_frame
        test_frame = test_frame_ori.copy()
        test_frame_rgb = FrameProcessor.get_frame_rgb(test_frame)
        test_detections = detector.detect_objects(test_frame_rgb)
        detector.draw_detections(test_frame, test_detections, color=(0, 255, 0))
        return test_frame

    @staticmethod
    def process_change_frame(base_frame, test_frame, detector):
        base_frame_rgb = FrameProcessor.get_frame_rgb(base_frame)
        test_frame_rgb = FrameProcessor.get_frame_rgb(test_frame)
        base_detections, test_detections, changes = FrameProcessor.detect_and_compare(base_frame_rgb, test_frame_rgb, detector)
        change_frame = test_frame.copy()
        FrameProcessor.draw_all_detections(change_frame, changes, detector)
        return change_frame

    @staticmethod
    def process_compass_frame(test_gps, test_compass, canvas):
        canvas[:, :, :] = 0
        gps, compass = test_gps, test_compass
        return FrameProcessor.draw_compass_gps(canvas, gps, compass)

    @staticmethod
    def draw_compass_gps(canvas, gps, compass):
        # Desenhar a posição e orientação do veículo no canvas
        x_in_map = int(gps[0] * 150) + canvas.shape[1] // 2
        y_in_map = canvas.shape[0] // 2 - int(gps[1] * 150) - canvas.shape[0] // 4
        cv2.circle(canvas, (x_in_map, y_in_map), 12, (0, 0, 255), 2)
        angle = np.arctan2(compass[1], compass[0]) - np.pi / 2
        nx_in_map = x_in_map + int(18 * np.cos(angle))
        ny_in_map = y_in_map + int(18 * np.sin(angle))
        cv2.line(canvas, (x_in_map, y_in_map), (nx_in_map, ny_in_map), (0, 255, 0), 1)
        return canvas

    @staticmethod
    def get_frame_rgb(frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    @staticmethod
    def detect_and_compare(base_frame, test_frame, detector):
        base_detections = detector.detect_objects(base_frame)
        test_detections = detector.detect_objects(test_frame)
        changes = detector.compare_detections(base_detections, test_detections)
        return base_detections, test_detections, changes

    @staticmethod
    def draw_all_detections(frame, changes, detector):
        detector.draw_detections(frame, changes['appeared'], color=(0, 255, 0), label_prefix='', isConfidence=False)
        detector.draw_detections(frame, changes['disappeared'], color=(0, 0, 255), label_prefix='Lost: ', isConfidence=False)
        detector.draw_detections(frame, changes['moved'], color=(0, 255, 255), label_prefix='Moved: ', isConfidence=False)

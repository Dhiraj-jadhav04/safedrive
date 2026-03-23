import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.spatial import distance as dist
from collections import deque

class DrowsinessDetector:
    def __init__(self, model_path='face_landmarker.task'):
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(self.options)
        
        # Landmark Indices (Standard Mediapipe Indices remain same)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Landmark Indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH_VERTICAL = [13, 14]   # Inner lip center points
        self.MOUTH_HORIZONTAL = [78, 308] # Inner corners
        self.FOREHEAD = [10, 67, 103, 108, 109, 151, 337, 338, 297, 332, 285]
        
        # Buffers
        self.ear_buffer = deque(maxlen=10)
        self.mar_buffer = deque(maxlen=10)
        self.rppg_buffer = deque(maxlen=150)
        
    def calculate_ear(self, landmarks, eye_indices, w, h):
        pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
        v1 = dist.euclidean(pts[1], pts[5])
        v2 = dist.euclidean(pts[2], pts[4])
        h1 = dist.euclidean(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h1)

    def calculate_mar(self, landmarks, v_indices, h_indices, w, h):
        v_pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in v_indices]
        h_pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in h_indices]
        vertical = dist.euclidean(v_pts[0], v_pts[1])
        horizontal = dist.euclidean(h_pts[0], h_pts[1])
        return vertical / (horizontal + 1e-6)

    def get_head_pose(self, landmarks, w, h):
        model_pts = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        img_pts = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[61].x * w, landmarks[61].y * h),
            (landmarks[291].x * w, landmarks[291].y * h)
        ], dtype="double")

        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1))
        _, rotation_vector, translation_vector = cv2.solvePnP(model_pts, img_pts, camera_matrix, dist_coeffs)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        # decomposeProjectionMatrix returns 7 values
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
        # euler_angles is usually a 2D array of [[pitch], [yaw], [roll]]
        return float(euler_angles[0][0]), float(euler_angles[1][0]), float(euler_angles[2][0])

    def estimate_bpm(self, frame, landmarks, w, h):
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.FOREHEAD])
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mean_val = cv2.mean(frame[:,:,1], mask=mask)[0]
        self.rppg_buffer.append(mean_val)
        if len(self.rppg_buffer) < 100: return 0
        signal = np.array(self.rppg_buffer)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        fft_res = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1.0/30)
        idx = np.where((freqs >= 0.75) & (freqs <= 3.5))
        if len(idx[0]) == 0: return 0
        return float(freqs[idx][np.argmax(fft_res[idx])] * 60)

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.detector.detect(mp_image)
        
        data = {"ear": 0.0, "mar": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0, "bpm": 0.0, "face_detected": False}
        
        if results.face_landmarks:
            data["face_detected"] = True
            lm = results.face_landmarks[0]
            data["ear"] = (self.calculate_ear(lm, self.LEFT_EYE, w, h) + self.calculate_ear(lm, self.RIGHT_EYE, w, h)) / 2
            data["mar"] = self.calculate_mar(lm, self.MOUTH_VERTICAL, self.MOUTH_HORIZONTAL, w, h)
            data["pitch"], data["yaw"], data["roll"] = self.get_head_pose(lm, w, h)
            data["bpm"] = self.estimate_bpm(frame, lm, w, h)
            
        return data, results

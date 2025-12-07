import cv2
import mediapipe as mp
import numpy as np
import av
import time
from math import hypot
from collections import deque
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- CONFIG HALAMAN ---
st.set_page_config(page_title="AI Focus Mobile", layout="centered")

st.title("📱 AI Focus Timer (Mobile Ver.)")
st.write("Arahkan kamera HP ke wajah Anda. Tekan 'Start' untuk memulai.")

# --- SIDEBAR CONFIG ---
pomodoro_minutes = st.sidebar.number_input("Durasi Fokus (Menit)", min_value=1, max_value=120, value=25)
threshold_pitch = st.sidebar.slider("Sensitivitas Menunduk", 5, 30, 10)
threshold_yaw = st.sidebar.slider("Sensitivitas Menoleh", 10, 45, 15)

# --- FUNGSI EAR (MATEMATIKA) ---
def calculate_ear(landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * img_w), int(lm.y * img_h)))
        
    v1 = hypot(coords[2][0] - coords[3][0], coords[2][1] - coords[3][1])
    v2 = hypot(coords[4][0] - coords[5][0], coords[4][1] - coords[5][1])
    h = hypot(coords[0][0] - coords[1][0], coords[0][1] - coords[1][1])
    
    if h == 0: return 0
    ear = (v1 + v2) / (2.0 * h)
    return ear

# --- KELAS PEMROSES VIDEO (PENGGANTI WHILE LOOP) ---
class FocusDetector(VideoTransformerBase):
    def __init__(self):
        # Setup MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            refine_landmarks=True 
        )
        
        # Variabel Logika
        self.is_calibrated = False
        self.normal_pitch = 0
        self.normal_yaw = 0
        self.ear_history = deque(maxlen=10)
        self.EAR_THRESHOLD = 0.15
        
        # Timer Logic
        self.start_time = None
        self.session_duration = pomodoro_minutes * 60
        self.timer_running = False
        self.time_left = self.session_duration

    def recv(self, frame):
        # 1. Konversi Format Gambar (WebRTC -> OpenCV)
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror effect (agar seperti cermin)
        img = cv2.flip(img, 1)
        img_h, img_w, _ = img.shape
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(image_rgb)
        
        # Default Status
        status_text = "WAJAH TIDAK TERDETEKSI"
        status_color = (0, 0, 255) # Merah
        is_focused_now = False
        
        # Inisialisasi Timer saat wajah pertama terdeteksi
        if self.start_time is None and results.multi_face_landmarks:
             self.start_time = time.time()
             self.timer_running = True

        # --- LOGIKA UTAMA (Sama seperti versi Desktop) ---
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_3d = []
                face_2d = []
                
                # Head Pose Logic
                key_indices = [33, 263, 1, 61, 291, 199]
                for idx in key_indices:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                raw_pitch = angles[0] * 360
                raw_yaw = angles[1] * 360

                # Auto Calibration
                if not self.is_calibrated:
                    self.normal_pitch = raw_pitch
                    self.normal_yaw = raw_yaw
                    self.is_calibrated = True

                rel_pitch = raw_pitch - self.normal_pitch
                rel_yaw = raw_yaw - self.normal_yaw
                
                # Eye Aspect Ratio (EAR)
                left_indices = [33, 133, 160, 144, 158, 153]
                right_indices = [362, 263, 385, 380, 387, 373]
                ear_left = calculate_ear(face_landmarks.landmark, left_indices, img_w, img_h)
                ear_right = calculate_ear(face_landmarks.landmark, right_indices, img_w, img_h)
                current_ear = (ear_left + ear_right) / 2.0
                
                self.ear_history.append(current_ear)
                if len(self.ear_history) > 0:
                    smooth_ear = sum(self.ear_history) / len(self.ear_history)
                else:
                    smooth_ear = 0.3

                # Penentuan Status
                is_sleeping = smooth_ear < self.EAR_THRESHOLD
                
                if is_sleeping:
                    status_text = "ALARM: TIDUR!"
                    status_color = (255, 0, 255) # Ungu
                    is_focused_now = False
                elif rel_yaw < -threshold_yaw:
                    status_text = "MENOLEH KIRI"
                    status_color = (0, 0, 255)
                    is_focused_now = False
                elif rel_yaw > threshold_yaw:
                    status_text = "MENOLEH KANAN"
                    status_color = (0, 0, 255)
                    is_focused_now = False
                elif rel_pitch < -threshold_pitch:
                    status_text = "MENUNDUK"
                    status_color = (0, 0, 255)
                    is_focused_now = False
                elif rel_pitch > 20: # Limit atas hardcoded
                    status_text = "MENDONGAK"
                    status_color = (0, 0, 255)
                    is_focused_now = False
                else:
                    status_text = "FOKUS"
                    status_color = (0, 255, 0) # Hijau
                    is_focused_now = True

        # --- UPDATE TIMER (Real-time Calculation) ---
        # Karena ini stream, kita pakai logic sederhana pengurangan waktu
        # Catatan: Ini logic sederhana, timer akan berhenti jika face tidak terdeteksi
        if is_focused_now and self.time_left > 0:
            # Kurangi sedikit waktu (asumsi 30fps, kurangi 1/30 detik)
            # Agar lebih akurat di webrtc, kita pakai time delta sederhana per frame
            self.time_left -= 0.05 # Estimasi kasar per frame processing
        
        if self.time_left <= 0:
            self.time_left = 0
            status_text = "SELESAI!"
            status_color = (0, 255, 255) # Kuning

        # --- DRAW UI ON FRAME ---
        # Hitung Menit:Detik
        minutes = int(self.time_left // 60)
        seconds = int(self.time_left % 60)
        timer_str = f"{minutes:02d}:{seconds:02d}"

        # Gambar Background Hitam Transparan di Atas
        cv2.rectangle(img, (0, 0), (img_w, 100), (0, 0, 0), -1)
        
        # Gambar Timer
        cv2.putText(img, "WAKTU TERSISA", (img_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, timer_str, (img_w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

        # Gambar Status
        cv2.putText(img, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MENJALANKAN WEBRTC ---
webrtc_streamer(
    key="focus-timer",
    video_processor_factory=FocusDetector,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}, # Audio dimatikan
)

st.info("Tips: Pastikan HP dan Laptop terhubung di WiFi yang sama. Akses menggunakan Network URL.")
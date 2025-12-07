import cv2
import mediapipe as mp
import numpy as np
import time
from math import hypot
from collections import deque
import winsound  # Library suara untuk Windows
import threading # Agar suara tidak bikin video lag

# --- 1. SETUP & INPUT DURASI ---
print("=== SETUP FOKUS TIMER ===")
try:
    user_input = input("Masukkan durasi sesi fokus (menit): ")
    POMODORO_MINUTES = float(user_input)
except ValueError:
    print("Input tidak valid, menggunakan default 25 menit.")
    POMODORO_MINUTES = 25

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    refine_landmarks=True 
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- FUNGSI AUDIO ASYNC (Agar video tidak patah-patah) ---
def play_sound(sound_type):
    """
    Menjalankan suara di thread terpisah.
    Tipe: 'start', 'pause', 'resume', 'alarm', 'warning'
    """
    def run_sound():
        try:
            if sound_type == 'start':
                # Nada naik (Tanda mulai)
                winsound.Beep(500, 150)
                winsound.Beep(1000, 300)
            elif sound_type == 'pause':
                # Nada turun (Tanda istirahat)
                winsound.Beep(700, 200)
                winsound.Beep(400, 200)
            elif sound_type == 'resume':
                # Nada naik pendek
                winsound.Beep(500, 150)
                winsound.Beep(800, 150)
            elif sound_type == 'alarm':
                # Alarm Nyaring (Tidur)
                winsound.Beep(2000, 300) 
            elif sound_type == 'warning':
                # Peringatan Rendah (Menunduk)
                winsound.Beep(600, 150)
        except Exception as e:
            pass # Abaikan error jika sistem sound bermasalah

    # Jalankan di thread baru
    threading.Thread(target=run_sound, daemon=True).start()

# --- CONFIG TIMER ---
time_left = POMODORO_MINUTES * 60
last_frame_time = time.time()
timer_running = False
is_paused_manual = False  

# --- VARIABEL KALIBRASI ---
normal_pitch = 0
normal_yaw = 0
is_calibrated = False

# --- CONFIG SENSITIVITAS GERAKAN KEPALA ---
THRESHOLD_PITCH_DOWN = 2
THRESHOLD_PITCH_UP = 20    
THRESHOLD_YAW = 15         

# --- CONFIG MENGANTUK (EAR) ---
EAR_THRESHOLD = 0.15       
ear_history = deque(maxlen=10) 

# --- CONFIG INTERVAL SUARA ---
# Agar alarm tidak bunyi berisik setiap mili-detik (spam)
last_alarm_time = 0
ALARM_COOLDOWN = 1.5 # Alarm hanya bunyi max setiap 1.5 detik

# Font & Warna
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)     
COLOR_RED = (0, 0, 255)       
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)  
COLOR_WHITE = (255, 255, 255)
COLOR_PURPLE = (255, 0, 255)
COLOR_ORANGE = (0, 165, 255)

# --- FUNGSI EAR ---
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

print(f"Sesi DIMULAI untuk {POMODORO_MINUTES} menit.")
print("Sound System Aktif.")
play_sound('start') # Bunyi awal

while cap.isOpened():
    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time

    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    status_text = "WAJAH TIDAK TERDETEKSI"
    status_color = COLOR_RED
    is_focused_now = False
    
    rel_pitch = 0
    rel_yaw = 0
    smooth_ear = 0.3 

    # --- LOGIKA UTAMA ---
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
            
            # 1. HEAD POSE
            key_indices = [33, 263, 1, 61, 291, 199]
            for idx in key_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
                if idx == 1: nose_2d = (x, y)

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

            if not is_calibrated:
                normal_pitch = raw_pitch
                normal_yaw = raw_yaw
                is_calibrated = True
                print(f"Kalibrasi Otomatis Selesai.")

            rel_pitch = raw_pitch - normal_pitch
            rel_yaw = raw_yaw - normal_yaw
            
            # 2. DETEKSI MATA (EAR)
            left_indices = [33, 133, 160, 144, 158, 153]
            right_indices = [362, 263, 385, 380, 387, 373]
            
            ear_left = calculate_ear(face_landmarks.landmark, left_indices, img_w, img_h)
            ear_right = calculate_ear(face_landmarks.landmark, right_indices, img_w, img_h)
            current_ear = (ear_left + ear_right) / 2.0
            
            ear_history.append(current_ear)
            smooth_ear = sum(ear_history) / len(ear_history)

            # 3. PENENTUAN STATUS & LOGIKA SUARA
            is_sleeping = smooth_ear < EAR_THRESHOLD
            
            # Cek apakah sudah waktunya membunyikan alarm (Cooldown system)
            time_since_alarm = current_time - last_alarm_time
            should_play_alarm = time_since_alarm > ALARM_COOLDOWN

            if is_paused_manual:
                status_text = "PAUSED (MANUAL)"
                status_color = COLOR_ORANGE
                is_focused_now = False 
            elif is_sleeping:
                status_text = "ALARM: TIDUR!"
                status_color = COLOR_PURPLE
                is_focused_now = False
                if should_play_alarm:
                    play_sound('alarm') # Bunyi Keras
                    last_alarm_time = current_time
            elif rel_yaw < -THRESHOLD_YAW:
                status_text = "MENOLEH KIRI"
                status_color = COLOR_RED
                is_focused_now = False
            elif rel_yaw > THRESHOLD_YAW:
                status_text = "MENOLEH KANAN"
                status_color = COLOR_RED
                is_focused_now = False
            elif rel_pitch < -THRESHOLD_PITCH_DOWN:
                status_text = "MENUNDUK"
                status_color = COLOR_RED
                is_focused_now = False
                if should_play_alarm:
                    play_sound('warning') # Bunyi Peringatan
                    last_alarm_time = current_time
            elif rel_pitch > THRESHOLD_PITCH_UP:
                status_text = "MENDONGAK"
                status_color = COLOR_RED
                is_focused_now = False
            else:
                status_text = "FOKUS"
                status_color = COLOR_GREEN
                is_focused_now = True

            # VISUALISASI HIDUNG
            nose_end_point2D, _ = cv2.projectPoints(np.array([(0.0, 0.0, 800.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(image, p1, p2, COLOR_BLUE, 3)

    # --- UPDATE TIMER ---
    if is_focused_now and not is_paused_manual and time_left > 0:
        time_left -= delta_time
        timer_running = True
    else:
        timer_running = False

    if time_left <= 0:
        time_left = 0
        status_text = "SELESAI!"
        status_color = COLOR_YELLOW
        # Bunyi 'Pause' jika timer selesai agar user sadar
        if not is_paused_manual:
            play_sound('pause')
        is_paused_manual = True 

    # --- UI RENDER ---
    cv2.rectangle(image, (0, 0), (img_w, 140), (0, 0, 0), -1)
    cv2.addWeighted(image, 0.7, image, 0.3, 0, image[0:140, 0:img_w])

    minutes = int(time_left // 60)
    seconds = int(time_left % 60)
    timer_text = f"{minutes:02d}:{seconds:02d}"
    
    if is_paused_manual:
        timer_col = COLOR_ORANGE
    elif is_focused_now:
        timer_col = COLOR_GREEN
    else:
        timer_col = COLOR_RED
    
    cv2.putText(image, "TIME LEFT", (img_w - 280, 40), FONT, 0.8, COLOR_WHITE, 1)
    cv2.putText(image, timer_text, (img_w - 280, 100), FONT, 2.5, timer_col, 4)

    cv2.putText(image, status_text, (30, 90), FONT, 1.2, status_color, 3)
    
    debug_text = f"P:{int(rel_pitch)} | Y:{int(rel_yaw)} | EAR:{smooth_ear:.2f}"
    cv2.putText(image, debug_text, (30, 30), FONT, 0.6, COLOR_YELLOW, 1)
    
    control_text = "'p': PAUSE | 's': RESUME | 'c': RE-CALIBRATE | 'q': QUIT"
    cv2.putText(image, control_text, (30, 130), FONT, 0.6, COLOR_WHITE, 1)

    cv2.imshow('AI Focus Timer V3.0 (Sound FX)', image)
    
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c'):
        is_calibrated = False
    elif key == ord('p'):
        if not is_paused_manual: # Hanya bunyi jika belum pause
            play_sound('pause')
        is_paused_manual = True
    elif key == ord('s'):
        if is_paused_manual: # Hanya bunyi jika sedang pause
            play_sound('resume')
        is_paused_manual = False

cap.release()
cv2.destroyAllWindows()
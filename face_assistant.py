import cv2
import mediapipe as mp
import numpy as np
import time

# --- 1. SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- CONFIG TIMER ---
POMODORO_MINUTES = 25
time_left = POMODORO_MINUTES * 60
last_frame_time = time.time()
timer_running = False

# --- VARIABEL KALIBRASI (PENTING!) ---
# Kita simpan posisi 'normal' user
normal_pitch = 0
normal_yaw = 0
is_calibrated = False

# --- CONFIG SENSITIVITAS RELATIF ---
# Berapa derajat penyimpangan dari posisi normal yang diizinkan?
# Kita perketat jadi 10 derajat saja.
THRESHOLD_PITCH_DOWN = 10  # Batas menunduk (Relative)
THRESHOLD_PITCH_UP = 20    # Batas mendongak (lebih santai)
THRESHOLD_YAW = 20         # Batas toleh

# Font & Warna
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)     
COLOR_RED = (0, 0, 255)       
COLOR_BLUE = (255, 0, 0)      
COLOR_YELLOW = (0, 255, 255)  
COLOR_WHITE = (255, 255, 255)

print("Sistem Berjalan. Duduk tegak untuk kalibrasi awal.")

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
    
    # Variabel raw pitch/yaw
    raw_pitch, raw_yaw = 0, 0
    
    # Variabel relative (setelah dikurangi kalibrasi)
    rel_pitch, rel_yaw = 0, 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
            
            key_indices = [33, 263, 1, 61, 291, 199]

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in key_indices:
                    if idx == 1: nose_2d = (lm.x * img_w, lm.y * img_h)
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

            # Raw Pitch/Yaw (Data mentah)
            raw_pitch = angles[0] * 360
            raw_yaw = angles[1] * 360

            # --- LOGIKA KALIBRASI ---
            # Jika belum kalibrasi, ambil frame pertama sebagai standar
            if not is_calibrated:
                normal_pitch = raw_pitch
                normal_yaw = raw_yaw
                is_calibrated = True
                print(f"Terkalibrasi! Normal Pitch: {normal_pitch}, Normal Yaw: {normal_yaw}")

            # Hitung Perbedaan (Relative) dari posisi Normal
            rel_pitch = raw_pitch - normal_pitch
            rel_yaw = raw_yaw - normal_yaw

            # --- LOGIKA STATUS ---
            # Jika rel_pitch NEGATIF besar, berarti menunduk dari posisi normal
            if rel_yaw < -THRESHOLD_YAW:
                status_text = "MENOLEH KIRI"
                status_color = COLOR_RED
                is_focused_now = False
            elif rel_yaw > THRESHOLD_YAW:
                status_text = "MENOLEH KANAN"
                status_color = COLOR_RED
                is_focused_now = False
            elif rel_pitch < -THRESHOLD_PITCH_DOWN: # Perhatikan tanda minus
                status_text = "MENUNDUK"
                status_color = COLOR_RED
                is_focused_now = False
            elif rel_pitch > THRESHOLD_PITCH_UP:
                status_text = "MENDONGAK"
                status_color = COLOR_RED
                is_focused_now = False
            else:
                status_text = "FOKUS"
                status_color = COLOR_GREEN
                is_focused_now = True

            # Visualisasi Garis Hidung
            nose_end_point2D, _ = cv2.projectPoints(np.array([(0.0, 0.0, 800.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(image, p1, p2, COLOR_BLUE, 3)

    # --- UPDATE TIMER ---
    if is_focused_now and time_left > 0:
        time_left -= delta_time
        timer_running = True
    else:
        timer_running = False

    if time_left <= 0:
        time_left = 0
        status_text = "SELESAI!"
        status_color = COLOR_YELLOW

    # --- UI RENDER ---
    # Background Bar
    cv2.rectangle(image, (0, 0), (img_w, 130), (0, 0, 0), -1)
    cv2.addWeighted(image, 0.7, image, 0.3, 0, image[0:130, 0:img_w])

    # Timer Text
    minutes = int(time_left // 60)
    seconds = int(time_left % 60)
    timer_text = f"{minutes:02d}:{seconds:02d}"
    timer_col = COLOR_GREEN if timer_running else COLOR_RED
    
    cv2.putText(image, "TIME LEFT", (img_w - 280, 40), FONT, 0.8, COLOR_WHITE, 1)
    cv2.putText(image, timer_text, (img_w - 280, 100), FONT, 2.5, timer_col, 4)

    # Status Text
    cv2.putText(image, status_text, (30, 90), FONT, 1.5, status_color, 3)
    
    # Info Kalibrasi (Kiri Atas)
    # Tampilkan Raw Pitch vs Relative Pitch agar Anda paham bedanya
    debug_text = f"Raw Pitch: {int(raw_pitch)} | Normal: {int(normal_pitch)} | Diff: {int(rel_pitch)}"
    cv2.putText(image, debug_text, (30, 30), FONT, 0.6, COLOR_WHITE, 1)
    cv2.putText(image, "Tekan 'C' untuk Kalibrasi Ulang", (30, 120), FONT, 0.5, COLOR_YELLOW, 1)

    cv2.imshow('AI Focus Assistant - Calibrated', image)
    
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c'): # Tombol Kalibrasi Manual
        is_calibrated = False # Reset flag agar frame berikutnya dianggap normal
        status_text = "KALIBRASI ULANG..."

cap.release()
cv2.destroyAllWindows()
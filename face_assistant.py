import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

# --- 1. KONFIGURASI SISTEM ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- 2. VARIABEL APLIKASI ---
POMODORO_MINUTES = 25
initial_time = POMODORO_MINUTES * 60
time_left = initial_time
last_frame_time = time.time()

# Statistik untuk Laporan Akhir
start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
total_focus_duration = 0 # Detik
distraction_count = 0

# Variabel Kalibrasi
normal_pitch = 0
normal_yaw = 0
is_calibrated = False

# Ambang Batas (Sensitivitas)
THRESHOLD_PITCH_DOWN = 10 # Batas menunduk (Relative)
THRESHOLD_PITCH_UP = 20   # Batas mendongak
THRESHOLD_YAW = 20        # Batas toleh kanan/kiri

# Warna & Font
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)     
COLOR_RED = (0, 0, 255)       
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)  
COLOR_WHITE = (255, 255, 255)

print(f"--- AI FOCUS ASSISTANT ---")
print(f"Sistem Berjalan. Tekan 'c' untuk Kalibrasi, 'q' untuk Keluar.")

while cap.isOpened():
    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time

    success, image = cap.read()
    if not success: break

    # Pre-processing
    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Default Status
    status_text = "WAJAH TIDAK TERDETEKSI"
    status_color = COLOR_RED
    is_focused_now = False
    
    # Variabel Visualisasi (Default)
    raw_pitch, raw_yaw = 0, 0
    rel_pitch = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
            
            # Ambil 6 Titik Kunci Wajah
            key_indices = [33, 263, 1, 61, 291, 199]

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in key_indices:
                    if idx == 1: nose_2d = (lm.x * img_w, lm.y * img_h)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # --- MATEMATIKA PNP (HEAD POSE) ---
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Konversi ke Derajat
            raw_pitch = angles[0] * 360
            raw_yaw = angles[1] * 360

            # --- LOGIKA KALIBRASI OTOMATIS ---
            if not is_calibrated:
                normal_pitch = raw_pitch
                normal_yaw = raw_yaw
                is_calibrated = True
                print(f"Kalibrasi Berhasil! Normal Pitch: {normal_pitch:.2f}")

            # Hitung Nilai Relatif (Selisih dari posisi normal)
            rel_pitch = raw_pitch - normal_pitch
            rel_yaw = raw_yaw - normal_yaw

            # --- LOGIKA STATUS ---
            if rel_yaw < -THRESHOLD_YAW:
                status_text = "MENOLEH KIRI"
                status_color = COLOR_RED
                is_focused_now = False
            elif rel_yaw > THRESHOLD_YAW:
                status_text = "MENOLEH KANAN"
                status_color = COLOR_RED
                is_focused_now = False
            elif rel_pitch < -THRESHOLD_PITCH_DOWN:
                status_text = "MENUNDUK (MAIN HP?)"
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

            # --- VISUALISASI PINOCCHIO ---
            nose_end_point2D, _ = cv2.projectPoints(np.array([(0.0, 0.0, 800.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(image, p1, p2, COLOR_BLUE, 3)

    # --- UPDATE TIMER & STATISTIK ---
    if is_focused_now and time_left > 0:
        time_left -= delta_time
        total_focus_duration += delta_time # Tambah statistik
    
    # Hitung jumlah distraksi (Logika transisi dari Fokus ke Tidak Fokus)
    # (Sederhana: Jika status merah muncul, kita anggap distraksi sedang terjadi)

    if time_left <= 0:
        time_left = 0
        status_text = "SESI SELESAI!"
        status_color = COLOR_YELLOW

    # --- UI RENDER (TAMPILAN) ---
    # Background Atas
    cv2.rectangle(image, (0, 0), (img_w, 130), (0, 0, 0), -1)
    cv2.addWeighted(image, 0.7, image, 0.3, 0, image[0:130, 0:img_w])

    # Timer
    minutes = int(time_left // 60)
    seconds = int(time_left % 60)
    timer_text = f"{minutes:02d}:{seconds:02d}"
    timer_col = COLOR_GREEN if is_focused_now else COLOR_RED
    
    cv2.putText(image, "TIMER", (img_w - 250, 40), FONT, 0.8, COLOR_WHITE, 1)
    cv2.putText(image, timer_text, (img_w - 280, 100), FONT, 2.5, timer_col, 4)

    # Status
    cv2.putText(image, status_text, (30, 90), FONT, 1.5, status_color, 3)
    
    # Info Teknis
    debug_text = f"Pitch Normal: {int(normal_pitch)} | Pitch Saat Ini: {int(raw_pitch)} | Selisih: {int(rel_pitch)}"
    cv2.putText(image, debug_text, (30, 30), FONT, 0.6, COLOR_WHITE, 1)
    
    # Instruksi Bawah
    cv2.putText(image, "Tekan 'c': Kalibrasi Ulang | Tekan 'q': Simpan & Keluar", (30, img_h - 30), FONT, 0.6, COLOR_YELLOW, 2)

    cv2.imshow('AI Focus Assistant - Final Portfolio', image)
    
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c'):
        is_calibrated = False
        status_text = "KALIBRASI..."

# --- GENERATE LAPORAN ---
print("Menyimpan Laporan...")
with open("laporan_fokus.txt", "w") as f:
    f.write("=== LAPORAN PRODUKTIVITAS AI ===\n")
    f.write(f"Waktu Mulai : {start_timestamp}\n")
    f.write(f"Waktu Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"--------------------------------\n")
    f.write(f"Total Waktu Fokus: {int(total_focus_duration // 60)} menit {int(total_focus_duration % 60)} detik\n")
    f.write(f"Persentase Fokus : {int((total_focus_duration / (initial_time - time_left + 0.1)) * 100)}%\n")
    f.write(f"--------------------------------\n")
    f.write("Good Job! Tetap Semangat.\n")

cap.release()
cv2.destroyAllWindows()
print(f"Laporan tersimpan di 'laporan_fokus.txt'")
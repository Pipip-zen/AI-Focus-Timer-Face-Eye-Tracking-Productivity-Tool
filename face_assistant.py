import cv2
import mediapipe as mp
import numpy as np
import time

# --- 1. KONFIGURASI & VARIABEL ---
# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Setup Kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Resolusi HD agar UI lebih lega
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- SETTING TIMER POMODORO ---
POMODORO_MINUTES = 25
time_left = POMODORO_MINUTES * 60 # Konversi ke detik
last_frame_time = time.time() # Untuk menghitung delta waktu
timer_running = False # Status apakah timer sedang jalan

# Ambang Batas (Threshold) Sudut Kepala
THRESHOLD_PITCH = 15
THRESHOLD_YAW = 20

# Font & Warna UI
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

print(f"Sistem Siap. Timer {POMODORO_MINUTES} menit. Tekan 'q' untuk keluar.")

while cap.isOpened():
    current_time = time.time()
    delta_time = current_time - last_frame_time # Waktu yang berlalu sejak frame sebelumnya
    last_frame_time = current_time

    success, image = cap.read()
    if not success:
        print("Kamera error.")
        break

    # Flip dan ubah ke RGB
    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Proses deteksi wajah
    results = face_mesh.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Default status jika wajah tidak terdeteksi
    status_text = "WAJAH TIDAK TERDETEKSI"
    status_color = COLOR_RED
    is_focused_now = False
    pitch, yaw = 0, 0 # Nilai default

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d = []
            face_2d = []
            
            # Titik kunci: Mata Kiri(33), Kanan(263), Hidung(1), Mulut Kiri(61), Kanan(291), Dagu(199)
            key_indices = [33, 263, 1, 61, 291, 199]

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in key_indices:
                    if idx == 1: # Simpan koordinat hidung untuk garis
                        nose_2d = (lm.x * img_w, lm.y * img_h)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # --- MATEMATIKA PNP (Head Pose) ---
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch = angles[0] * 360
            yaw = angles[1] * 360

            # --- LOGIKA STATUS FOKUS ---
            if yaw < -THRESHOLD_YAW:
                status_text = "MENOLEH KIRI - PAUSED"
                status_color = COLOR_RED
                is_focused_now = False
            elif yaw > THRESHOLD_YAW:
                status_text = "MENOLEH KANAN - PAUSED"
                status_color = COLOR_RED
                is_focused_now = False
            elif pitch < -THRESHOLD_PITCH:
                status_text = "MENUNDUK - PAUSED"
                status_color = COLOR_RED
                is_focused_now = False
            elif pitch > THRESHOLD_PITCH:
                status_text = "MENDONGAK - PAUSED"
                status_color = COLOR_RED
                is_focused_now = False
            else:
                status_text = "FOKUS - TIMER BERJALAN"
                status_color = COLOR_GREEN
                is_focused_now = True

            # Visualisasi Garis Hidung (Opsional, agar terlihat canggih)
            nose_end_point2D, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(image, p1, p2, COLOR_YELLOW, 2)

    # --- UPDATE LOGIKA TIMER ---
    if is_focused_now and time_left > 0:
        time_left -= delta_time # Kurangi waktu hanya jika fokus
        timer_running = True
    else:
        timer_running = False # Timer pause

    if time_left <= 0:
        time_left = 0
        status_text = "SESI SELESAI! ISTIRAHAT."
        status_color = COLOR_YELLOW
        timer_running = False

    # --- RENDER UI (TAMPILAN) ---
    # Format waktu MM:SS
    minutes = int(time_left // 60)
    seconds = int(time_left % 60)
    timer_text = f"{minutes:02d}:{seconds:02d}"
    
    # Latar belakang semi-transparan di atas agar tulisan terbaca
    cv2.rectangle(image, (0, 0), (img_w, 120), (0, 0, 0), -1)
    cv2.addWeighted(image, 0.7, image, 0.3, 0, image[0:120, 0:img_w]) # Efek transparansi gelap

    # Tampilkan Timer Besar
    timer_color = COLOR_GREEN if timer_running else COLOR_RED
    if time_left == 0: timer_color = COLOR_YELLOW
    cv2.putText(image, timer_text, (img_w - 300, 90), FONT, 3, timer_color, 5)
    cv2.putText(image, "TIME LEFT", (img_w - 300, 30), FONT, 0.8, COLOR_WHITE, 2)

    # Tampilkan Status & Data
    cv2.putText(image, status_text, (30, 80), FONT, 1.2, status_color, 3)
    # Data teknis kecil di pojok kiri atas
    cv2.putText(image, f"Pitch: {int(pitch)}  Yaw: {int(yaw)}", (30, 30), FONT, 0.6, COLOR_WHITE, 1)

    cv2.imshow('AI Focus Assistant - Final', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
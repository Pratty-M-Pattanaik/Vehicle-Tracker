import cv2
import sys
import numpy as np
import time

# CHOOSE THE REQUIRED VIDEO HERE
video_path = "car_inter.mp4" 

# SETUP AND VIDEO LOADING
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"\n[ERROR]: Could not open '{video_path}'.")
    sys.exit()

for _ in range(2):
    ret, frame = cap.read()

# TARGET SELECTION 
window_name = f"SELECT TARGET FOR {video_path.upper()}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\n" + "="*60)
print("--> TRACKING ACTIVE:")
print("Drag a tight box around the vehicle you want, then press ENTER.")
print("="*60 + "\n")

while True:
    bbox = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    if bbox[2] > 0 and bbox[3] > 0:
        break
    else:
        cv2.waitKey(0)

cv2.destroyWindow(window_name)

def reset_tracker(img, cx, cy, w, h):
    new_x = int(cx - w / 2)
    new_y = int(cy - h / 2)
    try:
        t = cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        t = cv2.TrackerCSRT_create()
    t.init(img, (new_x, new_y, w, h))
    return t

ORIGINAL_W = int(bbox[2])
ORIGINAL_H = int(bbox[3])

tracker = reset_tracker(frame, bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2, ORIGINAL_W, ORIGINAL_H)

# KALMAN FILTER
kf = cv2.KalmanFilter(4, 2)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.001 
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0 
kf.errorCovPost = np.eye(4, dtype=np.float32)

init_cx = bbox[0] + bbox[2] / 2
init_cy = bbox[1] + bbox[3] / 2
kf.statePost = np.array([[init_cx], [init_cy], [0], [0]], dtype=np.float32)

cooldown_counter = 0
is_occluded = False
prev_time = 0

# LIVE TRACKING & FUSION LOOP
print("\n[STATUS]: Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    
    # CALCULATE FPS
    new_time = time.time()
    fps = 1 / (new_time - prev_time) if (new_time - prev_time) > 0 else 0
    prev_time = new_time
    
    # PREDICT
    prediction = kf.predict()
    pred_cx, pred_cy = prediction[0][0], prediction[1][0]
    
    # UPDATE 
    track_success, current_bbox = tracker.update(frame)
    
    # ANOMALY DETECTION
    forced_occlusion = False
    if track_success:
        raw_x, raw_y, raw_w, raw_h = [int(v) for v in current_bbox]
        tracker_cx = raw_x + raw_w / 2
        tracker_cy = raw_y + raw_h / 2
        jump_distance = np.sqrt((tracker_cx - pred_cx)**2 + (tracker_cy - pred_cy)**2)
        if jump_distance > 15.0:
            forced_occlusion = True
            is_occluded = True
            cooldown_counter = 30  
    
    if is_occluded:
        cooldown_counter -= 1
        if cooldown_counter <= 0:
            is_occluded = False
            tracker = reset_tracker(frame, pred_cx, pred_cy, ORIGINAL_W, ORIGINAL_H)
            
    # FUSION LOGIC
    if track_success and not forced_occlusion and not is_occluded:
        w, h = ORIGINAL_W, ORIGINAL_H
        x = int(tracker_cx - w / 2)
        y = int(tracker_cy - h / 2)
        kf.correct(np.array([[np.float32(tracker_cx)], [np.float32(tracker_cy)]]))
        source = "Object Tracker (Size Locked)"
        t_conf, m_conf = 1.00, 0.70
        color = (0, 255, 0)
    else:
        w, h = ORIGINAL_W, ORIGINAL_H
        x = int(pred_cx - w / 2)
        y = int(pred_cy - h / 2)
        source = "Motion Model (Constant Velocity)"
        t_conf, m_conf = 0.00, 0.95
        color = (0, 165, 255)
    
    # DRAWING
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.putText(frame, f"FPS: {int(fps)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Source: {source}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"BBox: [x:{x}, y:{y}, w:{w}, h:{h}]", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Tracker Conf: {t_conf:.2f}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Motion Conf: {m_conf:.2f}", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Fused Vehicle Tracker", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import cvzone
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
import torch
import time
import csv
import numpy as np

# ---------------- CONFIG ----------------
VIDEO_PATH = "data/videos/people.mp4"
CONF_THRESH = 0.5
ZONE_Y_TOP = 360
ZONE_Y_BOTTOM = 430
LOITER_THRESHOLD = 5.0
MAX_ALERTS = 6

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)

# ---------------- MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("models/yolov8n.pt").to(device)

# ---------------- TRACKER ----------------
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    embedder="mobilenet"
)

# ---------------- STATE ----------------
track_history = defaultdict(list)
track_start_pos = {}
track_zone_state = {}
zone_entry_time = {}
loitering_ids = set()

entries = 0
exits = 0
alerts = deque(maxlen=MAX_ALERTS)

# ---------------- EVENT LOGGER ----------------
csv_file = open("events.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "track_id", "event", "zone"])

def log_event(track_id, event):
    csv_writer.writerow([time.time(), track_id, event, "restricted_zone"])
    csv_file.flush()
    alerts.appendleft(f"ID {track_id} {event}")

# ---------------- LOOP ----------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (960, 540))
    h, w, _ = frame.shape

    # ---------------- DETECTION ----------------
    detections = []
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != 0 or conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # ---------------- TRACKING ----------------
    tracks = tracker.update_tracks(detections, frame=frame)
    current_time = time.time()

    active_ids = set()
    movement_map = np.zeros((h, w, 3), dtype=np.uint8)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        active_ids.add(track_id)

        l, t, r, b = map(int, track.to_ltrb())
        cx = (l + r) // 2
        cy = b

        if track_id not in track_start_pos:
            track_start_pos[track_id] = (cx, cy)

        track_history[track_id].append((cx, cy))
        history = track_history[track_id]

        # ---------------- ZONE LOGIC ----------------
        inside_zone = ZONE_Y_TOP <= cy <= ZONE_Y_BOTTOM
        prev_state = track_zone_state.get(track_id, False)

        if inside_zone and not prev_state:
            track_zone_state[track_id] = True
            zone_entry_time[track_id] = current_time
            entries += 1
            log_event(track_id, "ENTERED")

        elif inside_zone and prev_state:
            dwell = current_time - zone_entry_time.get(track_id, current_time)
            if dwell >= LOITER_THRESHOLD and track_id not in loitering_ids:
                loitering_ids.add(track_id)
                log_event(track_id, "LOITERING")

        elif not inside_zone and prev_state:
            track_zone_state[track_id] = False
            zone_entry_time.pop(track_id, None)
            exits += 1
            log_event(track_id, "EXITED")

        # ---------------- DRAW VIDEO ----------------
        label = str(track_id)
        if track_id in loitering_ids:
            label += " | LOITER"

        cvzone.cornerRect(frame, (l, t, r - l, b - t), l=8)
        cvzone.putTextRect(frame, label, (l, t - 10), scale=1, thickness=2)

        # ---------------- MOVEMENT MAP ----------------
        for i in range(1, len(history)):
            x1, y1 = history[i - 1]
            x2, y2 = history[i]
            cv2.line(movement_map, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.circle(movement_map, (cx, cy), 6, (0, 255, 255), -1)

        if len(history) >= 2:
            px, py = history[-2]
            cv2.arrowedLine(
                movement_map,
                (px, py),
                (cx, cy),
                (255, 255, 0),
                2,
                tipLength=0.4
            )

        # Start-point ID box
        sx, sy = track_start_pos[track_id]
        cv2.rectangle(
            movement_map,
            (sx - 12, sy - 20),
            (sx + 12, sy - 4),
            (60, 60, 60),
            -1
        )
        cv2.putText(
            movement_map,
            str(track_id),
            (sx - 6, sy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

    # ---------------- DRAW ZONE ----------------
    cv2.rectangle(frame, (0, ZONE_Y_TOP), (w, ZONE_Y_BOTTOM), (0, 0, 255), 2)

    # ---------------- DASHBOARD ----------------
    DASH_H, DASH_W = 700, 1500
    dashboard = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)
    dashboard[:] = (20, 20, 20)

    # Place movement map on LEFT
    MAP_X, MAP_Y = 40, 80
    dashboard[MAP_Y:MAP_Y+h, MAP_X:MAP_X+w] = movement_map

    # RIGHT PANEL
    RIGHT_X = MAP_X + w + 60
    RIGHT_Y = MAP_Y
    GAP = 32

    cv2.putText(
        dashboard,
        "BEHAVIOR ANALYSIS",
        (RIGHT_X, RIGHT_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    stats = [
        f"Active IDs: {len(active_ids)}",
        f"Entries: {entries}",
        f"Exits: {exits}",
        f"Loitering: {len(loitering_ids)}"
    ]

    for i, s in enumerate(stats):
        cv2.putText(
            dashboard,
            s,
            (RIGHT_X, RIGHT_Y + 40 + i * GAP),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    alert_y = RIGHT_Y + 40 + len(stats) * GAP + 40

    cv2.putText(
        dashboard,
        "ALERTS",
        (RIGHT_X, alert_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    for i, alert in enumerate(alerts):
        cv2.putText(
            dashboard,
            alert,
            (RIGHT_X, alert_y + 40 + i * GAP),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # ---------------- SHOW WINDOWS ----------------
    cv2.imshow("Video Feed", frame)
    cv2.imshow("Analytics Dashboard", dashboard)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
csv_file.close()
cap.release()
cv2.destroyAllWindows()

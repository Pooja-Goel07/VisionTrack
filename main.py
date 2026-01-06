from ultralytics import YOLO
import cv2
import cvzone
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import torch
import time
import csv
import numpy as np

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture("data/videos/people.mp4")

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
track_zone_state = {}
zone_entry_time = {}
loitering_ids = set()

# ---------------- TOGGLES ----------------
show_tracks = False  # press 't'
show_panel = True    # press 'p'

# ---------------- ANALYTICS ----------------
stats = {
    "active_ids": set(),
    "entries": 0,
    "exits": 0,
    "loitering": set()
}

recent_alerts = []
MAX_ALERTS = 6
PANEL_WIDTH = 300

# ---------------- EVENT LOGGER ----------------
csv_file = open("events.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "track_id", "event", "zone"])

def log_event(track_id, event):
    timestamp = time.time()
    csv_writer.writerow([timestamp, track_id, event, "restricted_zone"])
    csv_file.flush()

    if event == "ENTRY":
        stats["entries"] += 1
        recent_alerts.append(f"ID {track_id} ENTERED")
    elif event == "EXIT":
        stats["exits"] += 1
        recent_alerts.append(f"ID {track_id} EXITED")
    elif event == "LOITERING":
        stats["loitering"].add(track_id)
        recent_alerts.append(f"⚠ ID {track_id} LOITERING")

    if len(recent_alerts) > MAX_ALERTS:
        recent_alerts.pop(0)

# ---------------- ZONE ----------------
ZONE_Y_TOP = 360
ZONE_Y_BOTTOM = 430
LOITER_THRESHOLD = 5.0  # seconds

# ---------------- LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (960, 540))

    # Reset per-frame stats
    stats["active_ids"].clear()

    # Draw zone
    cv2.rectangle(
        img,
        (0, ZONE_Y_TOP),
        (img.shape[1], ZONE_Y_BOTTOM),
        (0, 0, 255),
        2
    )

    results = model(img, stream=True)
    detections = []

    # ---------------- DETECTION ----------------
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls != 0 or conf < 0.5:
                continue

            detections.append(([x1, y1, w, h], conf, "person"))

    # ---------------- TRACKING ----------------
    tracks = tracker.update_tracks(detections, frame=img)
    current_time = time.time()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        stats["active_ids"].add(track_id)

        l, t, r, b = map(int, track.to_ltrb())
        w, h = r - l, b - t

        cx = l + w // 2
        cy = b  # bottom-center

        track_history[track_id].append((cx, cy))

        inside_zone = ZONE_Y_TOP <= cy <= ZONE_Y_BOTTOM
        prev_state = track_zone_state.get(track_id, False)

        label = f"ID {track_id}"

        # -------- ENTRY / EXIT / LOITER --------
        if inside_zone:
            if not prev_state:
                track_zone_state[track_id] = True
                zone_entry_time[track_id] = current_time
                log_event(track_id, "ENTRY")
            else:
                dwell_time = current_time - zone_entry_time.get(track_id, current_time)
                if dwell_time >= LOITER_THRESHOLD and track_id not in loitering_ids:
                    loitering_ids.add(track_id)
                    log_event(track_id, "LOITERING")
                if track_id in loitering_ids:
                    label += " | LOITERING"
        else:
            if prev_state:
                track_zone_state[track_id] = False
                zone_entry_time.pop(track_id, None)
                log_event(track_id, "EXIT")

        # ---------------- DRAW ----------------
        cvzone.cornerRect(img, (l, t, w, h), l=8)
        cvzone.putTextRect(img, label, (l, t - 10), scale=1, thickness=2)

        if show_tracks:
            pts = track_history[track_id]
            for i in range(1, len(pts)):
                cv2.line(img, pts[i - 1], pts[i], (200, 0, 200), 2)

    # ---------------- SIDE PANEL ----------------
    if show_panel:
        panel = np.zeros((img.shape[0], PANEL_WIDTH, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        cv2.putText(panel, "ANALYTICS", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        y = 80
        gap = 35

        cv2.putText(panel, f"Active IDs: {len(stats['active_ids'])}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += gap

        cv2.putText(panel, f"Entries: {stats['entries']}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += gap

        cv2.putText(panel, f"Exits: {stats['exits']}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += gap

        cv2.putText(panel, f"Loitering: {len(stats['loitering'])}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.putText(panel, "ALERTS", (20, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        ay = y + 90
        for alert in recent_alerts:
            cv2.putText(panel, alert, (20, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            ay += 25

        final_frame = cv2.hconcat([img, panel])
    else:
        final_frame = img

    cv2.imshow("Behavior Analysis System", final_frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    if key == ord('t'):
        show_tracks = not show_tracks
    if key == ord('p'):
        show_panel = not show_panel

# ---------------- CLEANUP ----------------
csv_file.close()
cap.release()
cv2.destroyAllWindows()

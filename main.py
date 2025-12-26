from ultralytics import YOLO
import cv2
import cvzone
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import torch

cap = cv2.VideoCapture("data/videos/people.mp4")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("models/yolov8n.pt").to(device)

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    embedder="mobilenet"
)

track_history = defaultdict(list)
track_zone_state = {}
entered_ids = set()
exited_ids = set()

classNames = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
    "chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
    "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

ZONE_Y_TOP = 360
ZONE_Y_BOTTOM = 430

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (960, 540))

    cv2.rectangle(
        img,
        (0, ZONE_Y_TOP),
        (img.shape[1], ZONE_Y_BOTTOM),
        (0, 0, 255),
        3
    )

    results = model(img, stream=True)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if classNames[cls] != "person":
                continue
            if conf < 0.5:
                continue

            detections.append(([x1, y1, w, h], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        w, h = r - l, b - t

        cx = l + w // 2
        cy = b

        track_history[track_id].append((cx, cy))

        inside_zone = ZONE_Y_TOP <= cy <= ZONE_Y_BOTTOM
        prev_state = track_zone_state.get(track_id, False)

        event_label = ""

        if inside_zone and not prev_state:
            track_zone_state[track_id] = True
            entered_ids.add(track_id)
            event_label = "ENTER"
            print(f"[ENTRY] ID {track_id}")

        elif not inside_zone and prev_state:
            track_zone_state[track_id] = False
            exited_ids.add(track_id)
            event_label = "EXIT"
            print(f"[EXIT] ID {track_id}")

        cvzone.cornerRect(img, (l, t, w, h), l=9)

        label = f"ID {track_id}"
        if event_label:
            label += f" | {event_label}"

        cvzone.putTextRect(
            img,
            label,
            (l, t - 10),
            scale=1.2,
            thickness=2,
            offset=3
        )

        points = track_history[track_id]
        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], (255, 0, 255), 2)

    cv2.putText(
        img,
        f"Entered: {len(entered_ids)}  Exited: {len(exited_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Zone-Based People Tracking", img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

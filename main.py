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
)

track_history = defaultdict(list)
MAX_HISTORY = 50

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

LINE_Y = 480

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (960, 540))

    cv2.line(img, (0, LINE_Y), (img.shape[1], LINE_Y), (0, 0, 255), 2)

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

        cvzone.cornerRect(img, (l, t, w, h), l=9)
        cvzone.putTextRect(
            img,
            f"ID {track_id}",
            (l, t - 10),
            scale=1.3,
            thickness=2,
            offset=3
        )

        points = track_history[track_id]
        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], (255, 0, 255), 2)

    cv2.imshow("People Tracking with History", img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

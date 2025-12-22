from ultralytics import YOLO
import cv2
import cvzone
import math
from deep_sort_realtime.deepsort_tracker import DeepSort

cap = cv2.VideoCapture("data/videos/people.mp4")

model = YOLO("models/yolov8n.pt")

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7
)

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

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detections = []

    # ------------------ YOLO detections ------------------
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if classNames[cls] != "person":
                continue
            if conf < 0.5:
                continue

            detections.append(([x1, y1, w, h], conf, "person"))

    # ------------------ DeepSORT tracking ------------------
    tracks = tracker.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        w, h = r - l, b - t

        cvzone.cornerRect(img, (l, t, w, h))
        cvzone.putTextRect(
            img,
            f"ID {track_id}",
            (l, t - 10),
            scale=1.5,
            thickness=2,
            offset=3
        )

    cv2.imshow("Image", img)
    cv2.waitKey(1)

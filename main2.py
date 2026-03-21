import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------------------
# Video source
# -------------------------------
# Set this to 0 for webcam or "video/crowd.mp4" for video file
video_source = 0
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Video not opened")
    exit()

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO("yolov8s.pt")  # small & fast

# -------------------------------
# Initialize Deep SORT tracker
# -------------------------------
tracker = DeepSort(
    max_age=30,
    n_init=1,
    embedder="mobilenet",
    half=True
)

# -------------------------------
# Monitoring square & threshold
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

boundary_x1, boundary_y1 = int(frame_width*0.1), int(frame_height*0.1)
boundary_x2, boundary_y2 = int(frame_width*0.9), int(frame_height*0.9)
crowd_threshold = 10

# -------------------------------
# Heatmap initialization
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
heatmap_decay = 0.95  # 0.95 = fade slowly, 0.9 = faster fade

# -------------------------------
# Data dictionary for table output
people_data = {}  # key=track_id, value={'entry':frame, 'last_seen':frame, 'inside_count':int}
frame_number = 0

# -------------------------------
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # Run YOLO detection
    results = model(frame)
    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            if int(classes[i]) == 0:  # class 0 = person
                x1, y1, x2, y2 = map(float, boxes[i])
                conf = float(confs[i])
                detections.append([[x1, y1, x2, y2], conf])

    # Update Deep SORT tracker
    tracks = tracker.update_tracks(detections, frame=frame) if len(detections) > 0 else []

    # Count people inside monitoring square
    people_inside = 0

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Track people data
        if track_id not in people_data:
            people_data[track_id] = {'entry': frame_number, 'last_seen': frame_number, 'inside_count': 0}
        else:
            people_data[track_id]['last_seen'] = frame_number

        # Check if person is inside monitoring square
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        if boundary_x1 < center_x < boundary_x2 and boundary_y1 < center_y < boundary_y2:
            people_inside += 1
            people_data[track_id]['inside_count'] += 1

        # Draw bounding box + stable ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Update heatmap
        heatmap[y1:y2, x1:x2] += 1

    # Apply heatmap decay
    heatmap *= heatmap_decay

    # Normalize heatmap for display
    heatmap_display = np.uint8(np.clip(heatmap*25, 0, 255))
    heatmap_display = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)

    # Overlay heatmap on frame
    overlay_frame = cv2.addWeighted(frame, 0.7, heatmap_display, 0.3, 0)

    # Draw monitoring square + people count
    cv2.rectangle(overlay_frame, (boundary_x1, boundary_y1), (boundary_x2, boundary_y2), (0, 255, 0), 2)
    cv2.putText(overlay_frame, f"People: {people_inside}", 
                (boundary_x1 + 10, boundary_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Crowd alert
    if people_inside >= crowd_threshold:
        cv2.putText(overlay_frame, f"CROWD ALERT! ({people_inside} people)", 
                    (boundary_x1 + 10, boundary_y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show frame
    cv2.imshow("Crowd Monitoring Robot", overlay_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# -------------------------------
# Save people data as CSV at the end
import pandas as pd
df = pd.DataFrame([
    {'ID': id,
        'Entry_Frame': v['entry'],
        'Last_Seen_Frame': v['last_seen'],
        'Frames_Inside_Square': v['inside_count']}
    for id, v in people_data.items()
])
df.to_csv("crowd_summary.csv", index=False)
print("Saved crowd summary table as 'crowd_summary.csv'")
print(df)

cap.release()
cv2.destroyAllWindows()

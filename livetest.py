from ultralytics import YOLO
import cv2
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load YOLO model ---
model = YOLO("best.pt") 

VIDEO_PATH = "test4.mp4"
pixel_to_meter = 0.05  # adjust based on calibration
FPS = 30  # approximate FPS
SMOOTH_WINDOW = 5
SPEED_LIMIT_KMPH = 60

vehicle_classes = ["ambulance","army vehicle","auto rickshaw","bicycle","bus","car","garbagevan",
                   "human hauler","minibus","minivan","motorbike","pickup","policecar","rickshaw",
                   "scooter","suv","taxi","three wheelers -CNG-","truck","van","wheelbarrow"]

# --- Data structures ---
vehicle_count = {cls:0 for cls in vehicle_classes}
prev_positions = {}      # track_id -> (cx, cy)
counted_ids = set()
speed_history = {}
report_data = defaultdict(list)

frame_counter = 0
cv2.namedWindow("Traffic Monitor", cv2.WINDOW_NORMAL)  # create resizable window

for frame_result in model.track(source=VIDEO_PATH, tracker="botsort.yaml", persist=True, stream=True):
    frame = frame_result.orig_img.copy()  # keep original resolution
    if frame is None:
        continue

    frame_counter += 1

    # Add right margin for counts
    margin = 250
    vis = cv2.copyMakeBorder(frame, 0, 0, 0, margin, cv2.BORDER_CONSTANT, value=[30,30,30])

    # Process detections
    for box in frame_result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        if label not in vehicle_classes or conf < 0.4:
            continue

        track_id = box.id
        if track_id is None:
            continue
        track_id = int(track_id)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        # Speed calculation
        if track_id in prev_positions:
            prev_cx, prev_cy = prev_positions[track_id]
            distance_pixels = math.hypot(cx - prev_cx, cy - prev_cy)
            speed_m_s = (distance_pixels * pixel_to_meter) * FPS
            speed_kmph = speed_m_s * 3.6
        else:
            speed_kmph = 0

        # Adaptive smoothing
        if track_id not in speed_history:
            speed_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
        speed_history[track_id].append(speed_kmph)
        smoothed_speed = sum(speed_history[track_id]) / len(speed_history[track_id])

        prev_positions[track_id] = (cx, cy)

        # Count vehicles once
        if track_id not in counted_ids:
            vehicle_count[label] += 1
            counted_ids.add(track_id)

        # Violation
        violation = "SPEEDING" if smoothed_speed > SPEED_LIMIT_KMPH else ""

        # Add to report
        report_data[track_id].append({
            "frame": frame_counter,
            "class": label,
            "speed_kmph": round(smoothed_speed,1),
            "violation": violation
        })

        # Draw boxes on vis
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(vis, f"{label} {conf:.2f} {smoothed_speed:.1f} km/h {violation}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Draw vehicle counts in the right margin
    y_pos = 30
    for cls, c in vehicle_count.items():
        cv2.putText(vis, f"{cls}: {c}", (frame.shape[1]+10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y_pos += 25

    # Display the frame with margin
    cv2.imshow("Traffic Monitor", vis)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()
all_rows = []
for track_id, entries in report_data.items():
    for e in entries:
        all_rows.append({
            "track_id": track_id,
            "class": e["class"],
            "speed_kmph": e["speed_kmph"],
            "violation": e["violation"]
        })

# --- Vehicle count per class ---
vehicle_counts = defaultdict(int)
for r in all_rows:
    vehicle_counts[r["class"]] += 1

plt.figure(figsize=(12,6))
sns.barplot(
    x=list(vehicle_counts.keys()), 
    y=list(vehicle_counts.values()), 
    palette="viridis",
    hue=None,
    dodge=False
)
plt.title("Vehicle Count per Class")
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- Average speed per vehicle class ---
speed_data = defaultdict(list)
for r in all_rows:
    speed_data[r["class"]].append(r["speed_kmph"])

avg_speeds = {cls: sum(vals)/len(vals) for cls, vals in speed_data.items()}

plt.figure(figsize=(12,6))
sns.barplot(
    x=list(avg_speeds.keys()),
    y=list(avg_speeds.values()),
    palette="coolwarm",
    hue=None,
    dodge=False
)
plt.title("Average Speed per Vehicle Class (km/h)")
plt.xticks(rotation=45)
plt.ylabel("Average Speed (km/h)")
plt.tight_layout()
plt.show()

# --- Traffic violations per vehicle class ---
violation_counts = defaultdict(int)
for r in all_rows:
    if r["violation"]:
        violation_counts[r["class"]] += 1

plt.figure(figsize=(12,6))
sns.barplot(
    x=list(violation_counts.keys()),
    y=list(violation_counts.values()),
    palette="rocket",
    hue=None,
    dodge=False
)
plt.title("Traffic Violations per Vehicle Class")
plt.xticks(rotation=45)
plt.ylabel("Violation Count")
plt.tight_layout()
plt.show()

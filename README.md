# realtime_traffic_monitor
Real-Time Traffic Monitoring using YOLOv8n &amp; OpenCV — detects cars, bikes, buses, and trucks from live video, counts vehicles per frame, and estimates their approximate speed using object tracking and basic physics. Trained with Kaggle traffic datasets for accurate detection and analysis.

## Overview

The model was trained using **Kaggle traffic datasets** to accurately detect multiple vehicle types in different lighting and road conditions.  
When running, it:
- Detects vehicles using YOLOv8n  
- Tracks each vehicle with a unique ID  
- Counts total vehicles per frame  
- Estimates approximate speed (in km/h) for each tracked object  

---

## Technologies Used

- **YOLOv8n** (Ultralytics)
- **OpenCV** for video processing
- **NumPy** for calculations
- **Python 3.8+**
- **Kaggle Datasets** for model training

---

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/real-time-traffic-monitor.git
   cd real-time-traffic-monitor
   Install dependencies:

pip install -r requirements.txt

python main.py

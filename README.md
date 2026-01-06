
### Real-Time Behavior Analysis System using YOLOv8 & DeepSORT
A real-time computer vision system for multi-person detection, tracking, and behavior analysis built using **YOLOv8 and DeepSORT**.

The system assigns persistent IDs to individuals and analyzes behaviors such as zone entry, exit, and loitering, supported by a live analytics dashboard and event logging.

### Key Features
Real-time person detection using YOLOv8
Multi-object tracking with persistent IDs via DeepSORT

Zone-based behavior analysis
- Entry detection
- Exit detection
- Loitering detection based on dwell time
  
Movement map visualization
- Trajectory paths
- Direction arrows
- Start-point ID markers
  
Live analytics dashboard
- Active tracked IDs
- Entry and exit counts
- Loitering count
- Real-time alerts

CSV-based event logging for offline analysis

### Device Configuration
The system automatically selects GPU (CUDA) if available; otherwise, it runs on CPU.

This ensures portability across different hardware setups without requiring code changes.
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### System Architecture
The application runs with two separate windows:
1. Video Feed
- Live person detection and tracking
- Bounding boxes, IDs, and restricted zone overlay
2. Analytics Dashboard
- Movement map (trajectory visualization)
- Behavior statistics
- Real-time alerts
  
This separation improves clarity and reflects real-world monitoring system design.

# Water Bottle Defect Detection System

A production-ready, real-time YOLO-based defect detection system for packaged water bottle production lines. The system uses an external USB webcam for live video capture and YOLOv8 for defect detection.

## Features

- **Real-time Detection**: Live video streaming with YOLO object detection
- **External USB Webcam Support**: Configurable camera index for external webcams
- **Defect Classification**: Detects cracks, scratches, missing labels, wrong labels, missing caps, and wrong cap colors
- **Web Dashboard**: Interactive web interface with live feed, analytics, and defect history
- **Database Logging**: MongoDB integration for defect storage and analysis
- **Dark Theme UI**: Modern, responsive dark theme interface

## System Architecture

### Components

1. **Camera Module** (`camera.py`): Handles external USB webcam initialization and frame capture
2. **YOLO Detector** (`yolo_detector.py`): YOLOv8 model loading, inference, and visualization
3. **Database Module** (`database.py`): MongoDB operations for defect logging and retrieval
4. **Flask Application** (`app.py`): Web server with MJPEG streaming and API endpoints
5. **Frontend**: HTML/CSS/JavaScript for live detection, dashboard, and history pages

### Technology Stack

- **Backend**: Python, Flask, OpenCV, Ultralytics YOLOv8
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Database**: MongoDB
- **Camera**: External USB Webcam (configurable index)

## Installation

### Prerequisites

1. **Python 3.8 or higher**
2. **MongoDB** (Install from https://www.mongodb.com/try/download/community)
3. **External USB Webcam**

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install MongoDB

**Windows:**
1. Download MongoDB Community Server from https://www.mongodb.com/try/download/community
2. Install and start MongoDB service
3. MongoDB will run on `localhost:27017` by default

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS (using Homebrew)
brew install mongodb-community
brew services start mongodb-community

# Start MongoDB
sudo systemctl start mongodb  # Linux
# or
mongod  # If not running as service
```

### Step 3: Configure External USB Webcam

## CRITICAL: External USB Webcam Setup

The system is configured to use an **EXTERNAL USB WEBCAM** (not your laptop's built-in camera).

### Finding Your Camera Index

1. **Connect your external USB webcam** to your computer

2. **List available cameras:**
   ```bash
   python camera.py
   ```
   
   Or run this Python command:
   ```bash
   python -c "import cv2; [print(f'Index {i}: Available') if cv2.VideoCapture(i).read()[0] else print(f'Index {i}: Not available') for i in range(5)]"
   ```

3. **Identify your external webcam:**
   - Index 0 is typically the built-in/laptop camera
   - Index 1, 2, 3, etc. are usually external USB webcams
   - Try opening each index to confirm which is your external webcam

4. **Update configuration:**
   Open `config.py` and set the `CAMERA_INDEX`:
   ```python
   CAMERA_INDEX = 1  # Change this to your external webcam index
   ```

### Testing Camera Access

Run this test script to verify your camera works:
```python
import cv2
from camera import Camera

cam = Camera(camera_index=1)  # Use your camera index
if cam.initialize():
    print("Camera initialized successfully!")
    frame = cam.get_frame()
    if frame is not None:
        print(f"Frame captured: {frame.shape}")
    cam.release()
else:
    print("Camera initialization failed!")
```

### Switching Between Webcams

1. Open `config.py`
2. Change `CAMERA_INDEX` to the desired camera index
3. Restart the application

### Troubleshooting Camera Issues

**Problem: Camera not detected**
- Ensure the external webcam is connected and powered on
- Try a different USB port
- Check if the camera works in other applications (e.g., Windows Camera app)
- Run `python camera.py` to list available cameras

**Problem: Camera index wrong**
- List all available cameras using the methods above
- Try different indices (1, 2, 3, etc.)
- Restart the application after changing the index

**Problem: Permission denied (Linux)**
- Add your user to the `video` group: `sudo usermod -a -G video $USER`
- Log out and log back in
- On Ubuntu, you may need to grant camera permissions in system settings

**Problem: Camera already in use**
- Close other applications using the camera
- Restart the application

## Configuration

Edit `config.py` to configure:

- **Camera Settings**:
  - `CAMERA_INDEX`: External USB webcam index (default: 1)
  - `CAMERA_WIDTH`, `CAMERA_HEIGHT`: Video resolution
  - `CAMERA_FPS`: Frame rate

- **YOLO Model**:
  - `MODEL_PATH`: Path to YOLO model (default: "yolov8n.pt" - will download automatically)
  - `CONFIDENCE_THRESHOLD`: Detection confidence threshold (0.0-1.0)
  - `IOU_THRESHOLD`: Non-maximum suppression threshold

- **Database**:
  - `MONGODB_URI`: MongoDB connection string
  - `DATABASE_NAME`: Database name
  - `COLLECTION_NAME`: Collection name

- **Defect Logging**:
  - `DEFECT_LOGGING_COOLDOWN`: Cooldown period to avoid duplicate logging (seconds)

## Running the Application

### Step 1: Start MongoDB

Ensure MongoDB is running:

```bash
# Windows: MongoDB should start automatically as a service
# Or start manually:
mongod

# Linux/macOS:
sudo systemctl start mongodb
# or
mongod
```

### Step 2: Run the Application

```bash
python app.py
```

The application will:
1. Initialize the external USB webcam
2. Load the YOLO model
3. Connect to MongoDB
4. Start the Flask web server

### Step 3: Access the Web Interface

Open your web browser and navigate to:

```
http://localhost:5000
```

## Usage

### Live Detection Page

1. Navigate to the **Live Detection** page (home page)
2. Click **Start Detection** to begin real-time defect detection
3. View live video feed with bounding boxes around detected defects
4. Monitor active defect count and total detected defects
5. Click **Stop Detection** to pause detection

### Analysis Dashboard

1. Navigate to the **Dashboard** page
2. View overall statistics:
   - Total bottles inspected
   - Total defects detected
   - Recent defects (24 hours)
3. View defect distribution by type (pie chart)
4. View defect trends over time (line chart)
5. Click **Refresh Data** to update statistics

### Defect History

1. Navigate to the **History** page
2. Browse all detected defects with thumbnail previews
3. Filter by defect type using the dropdown
4. Click on a defect card to view full-size image and details
5. Use pagination to browse through defects

## YOLO Model Configuration

### Using Pretrained Model (Demo)

The system uses YOLOv8n (nano) by default, which will download automatically on first run. This is suitable for demonstration purposes.

**Note**: The pretrained YOLOv8 model is trained on COCO dataset and may not detect water bottle defects accurately. For production use, you need a custom-trained model.

### Training a Custom Model

To train a custom YOLO model for water bottle defects:

1. **Collect Dataset**:
   - Gather images of water bottles with various defects
   - Include images of normal bottles as well
   - Aim for at least 100-200 images per defect class

2. **Annotate Images**:
   - Use annotation tools like LabelImg, CVAT, or Roboflow
   - Annotate bounding boxes for each defect
   - Classes: crack, scratch, missing_label, wrong_label, missing_cap, wrong_cap_color

3. **Organize Dataset** (YOLO format):
   ```
   dataset/
     train/
       images/
       labels/
     val/
       images/
       labels/
     data.yaml
   ```

4. **Train Model**:
   ```python
   from ultralytics import YOLO
   
   # Load pretrained model
   model = YOLO('yolov8n.pt')
   
   # Train
   model.train(
       data='dataset/data.yaml',
       epochs=100,
       imgsz=640,
       batch=16,
       name='bottle_defects'
   )
   ```

5. **Use Trained Model**:
   - After training, use `runs/detect/bottle_defects/weights/best.pt`
   - Update `MODEL_PATH` in `config.py` to point to your trained model

6. **Update Class Mapping**:
   - Ensure `DEFECT_CLASSES` in `config.py` matches your training classes

## API Endpoints

### Detection Control

- `POST /api/detection/start`: Start defect detection
- `POST /api/detection/stop`: Stop defect detection
- `GET /api/detection/status`: Get detection status and statistics

### Data Retrieval

- `GET /api/stats`: Get defect statistics and time series data
- `GET /api/defects?limit=100&skip=0&type=<defect_type>`: Get defect history
- `GET /api/camera/list`: List available cameras

### Video Streaming

- `GET /video_feed`: MJPEG video stream

## Project Structure

```
.
├── app.py                  # Flask application (main entry point)
├── camera.py               # External USB webcam handling
├── yolo_detector.py        # YOLO model and inference
├── database.py             # MongoDB operations
├── config.py               # Configuration file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/             # HTML templates
│   ├── base.html
│   ├── index.html         # Live detection page
│   ├── dashboard.html     # Analytics dashboard
│   └── history.html       # Defect history page
└── static/                # Static assets
    ├── css/
    │   └── style.css      # Dark theme styles
    └── js/
        └── main.js        # Common JavaScript
```

## Troubleshooting

### Application Won't Start

**Issue**: Camera initialization failed
- **Solution**: Check camera index in `config.py`, ensure external webcam is connected, try different USB port

**Issue**: MongoDB connection failed
- **Solution**: Ensure MongoDB is running (`mongod`), check connection string in `config.py`

**Issue**: YOLO model not found
- **Solution**: Model will download automatically on first run. Ensure internet connection is available.

### Video Feed Not Showing

**Issue**: Black screen or "Waiting for camera"
- **Solution**: 
  - Verify camera index is correct
  - Check if camera is being used by another application
  - Restart the application
  - Check camera permissions (Linux/macOS)

**Issue**: Video feed is from wrong camera
- **Solution**: Change `CAMERA_INDEX` in `config.py` and restart

### Detection Not Working

**Issue**: No defects detected
- **Solution**: 
  - The pretrained YOLOv8 model may not detect water bottle defects
  - Train a custom model with your specific defect classes
  - Lower confidence threshold in `config.py`
  - Ensure good lighting and camera positioning

**Issue**: Too many false positives
- **Solution**: 
  - Increase confidence threshold in `config.py`
  - Train a custom model with more diverse data
  - Adjust camera angle and lighting

### Database Issues

**Issue**: Defects not being logged
- **Solution**: 
  - Verify MongoDB is running
  - Check MongoDB connection in `config.py`
  - Check database logs for errors

**Issue**: Database connection timeout
- **Solution**: 
  - Ensure MongoDB is accessible at configured URI
  - Check firewall settings
  - Verify MongoDB service is running

## Performance Optimization

- **Model Size**: Use YOLOv8n (nano) for faster inference, YOLOv8x for better accuracy
- **Resolution**: Lower camera resolution for better FPS
- **Frame Rate**: Adjust `STREAM_FPS` in `config.py` to reduce bandwidth
- **GPU**: Install PyTorch with CUDA support for GPU acceleration (optional)

## License

This project is provided as-is for educational and demonstration purposes.

## Support

For issues related to:
- **YOLO**: https://github.com/ultralytics/ultralytics
- **OpenCV**: https://opencv.org/
- **MongoDB**: https://www.mongodb.com/docs/
- **Flask**: https://flask.palletsprojects.com/

## Notes

- The system uses a pretrained YOLOv8 model by default, which may not accurately detect water bottle defects
- For production use, train a custom YOLO model with your specific defect classes
- Ensure proper lighting and camera positioning for best results
- The external USB webcam configuration is critical - always verify the camera index before use

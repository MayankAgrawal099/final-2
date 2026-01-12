# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Start MongoDB

**Windows:**
- MongoDB should start automatically as a service
- Or run: `mongod`

**Linux/macOS:**
```bash
sudo systemctl start mongodb
# or
mongod
```

## 3. Configure External USB Webcam

1. Connect your external USB webcam
2. Find your camera index:
   ```bash
   python camera.py
   ```
3. Edit `config.py` and set `CAMERA_INDEX = 1` (or your webcam index)

## 4. Run the Application

```bash
python app.py
```

## 5. Open Web Browser

Navigate to: `http://localhost:5000`

## 6. Start Detection

1. Go to "Live Detection" page
2. Click "Start Detection" button
3. View real-time defect detection

## Notes

- The pretrained YOLOv8 model may not detect water bottle defects accurately
- For production, train a custom YOLO model with your defect classes
- Always verify your external webcam index before use

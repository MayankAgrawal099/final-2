# Troubleshooting Guide

## Issue: Camera Detected But Not Initializing / Localhost Not Starting

### Problem
- External webcam is detected but initialization fails
- Flask server (localhost) doesn't start

### Solution Applied
The code has been updated to make initialization **non-blocking**. The Flask server will now start even if:
- Camera initialization fails
- YOLO model loading fails
- Database connection fails

### What Changed
1. **Camera Initialization**: Now non-blocking - Flask starts even if camera fails
2. **YOLO Model Loading**: Errors are logged but don't prevent Flask from starting
3. **Database Connection**: Already was non-blocking (just shows warnings)

### How to Verify

1. **Test Camera Separately:**
   ```bash
   python test_camera.py
   ```
   This will verify your camera works independently.

2. **Run the Application:**
   ```bash
   python app.py
   ```
   
   Even if you see errors like:
   - "Camera initialization failed!"
   - "YOLO detector initialization failed!"
   
   The Flask server should still start and show:
   - "Starting Flask server at http://0.0.0.0:5000"
   - "Open your browser and navigate to the URL above"

3. **Access the Web Interface:**
   - Open browser to: `http://localhost:5000`
   - If camera failed, you'll see "Camera not initialized" message
   - Other features (dashboard, history) should still work

### If Camera Still Fails

1. **Check Camera Index:**
   - Run: `python camera.py`
   - Note which cameras are found
   - Update `CAMERA_INDEX` in `config.py`

2. **Check if Camera is in Use:**
   - Close other applications using the camera (Zoom, Teams, etc.)
   - Try restarting the application

3. **Check Permissions (Linux/Mac):**
   - Make sure you have camera permissions
   - Try running with sudo (not recommended for production)

4. **Try Different Camera Backend (Windows):**
   - Some cameras work better with DirectShow backend
   - You can try: `cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)`

### Common Error Messages

**"Camera initialization failed!"**
- Camera is detected but can't be opened
- Usually means camera is in use or permissions issue
- **Solution**: Flask will still start, but detection features won't work

**"Camera opened but cannot read frames"**
- Camera opens but can't read video frames
- Usually means camera hardware issue or driver problem
- **Solution**: Try different USB port, update camera drivers

**"YOLO detector initialization failed!"**
- Model file not found or download failed
- First run will download model automatically (needs internet)
- **Solution**: Check internet connection, ensure enough disk space

### Current Behavior

- ✅ Flask server **WILL START** even if camera fails
- ✅ Web interface **WILL BE ACCESSIBLE** at localhost:5000
- ⚠️ Detection features **WON'T WORK** if camera/detector failed
- ⚠️ Video feed will show "Camera not initialized" message
- ✅ Dashboard and History pages will still work (if database connected)

"""
Flask application for YOLO-based defect detection system.
Main application file with routes, video streaming, and API endpoints.
"""

import logging
import threading
import time
from datetime import datetime
from io import BytesIO
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from werkzeug.serving import WSGIRequestHandler

import config
from camera import Camera
from yolo_detector import YOLODetector
from database import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder=config.TEMPLATES_DIR, static_folder=config.STATIC_DIR)

# Global variables for detection system
camera = None
detector = None
db = None
detection_active = False
detection_thread = None
current_frame = None
frame_lock = threading.Lock()
detection_stats = {
    "total_detected": 0,
    "current_defects": 0,
    "last_detection_time": {}
}

# Cooldown tracking for defect logging
defect_log_cooldown = {}


def init_system():
    """Initialize camera, detector, and database."""
    global camera, detector, db
    
    # Initialize camera (non-blocking - allow Flask to start even if camera fails)
    logger.info("Initializing camera...")
    camera = Camera()
    try:
        if not camera.initialize():
            logger.error("Camera initialization failed! Server will start but detection will be unavailable.")
            logger.error("Please check camera connection and CAMERA_INDEX in config.py")
            # Don't return False - allow Flask to start
    except Exception as e:
        logger.error(f"Camera initialization error: {str(e)}")
        logger.error("Server will start but camera features will be unavailable.")
    
    # Initialize YOLO detector
    logger.info("Initializing YOLO detector...")
    detector = YOLODetector()
    try:
        if not detector.load_model():
            logger.error("YOLO detector initialization failed! Server will start but detection will be unavailable.")
            # Don't return False - allow Flask to start
    except Exception as e:
        logger.error(f"YOLO detector initialization error: {str(e)}")
        logger.error("Server will start but detection features will be unavailable.")
    
    # Initialize database (non-blocking)
    logger.info("Initializing database...")
    db = Database()
    if not db.connect():
        logger.warning("Database connection failed. Defect logging will be disabled.")
        logger.warning("Please start MongoDB: mongod")
    
    logger.info("System initialization completed! Flask server starting...")
    return True  # Always return True to allow Flask to start


def detection_loop():
    """Main detection loop running in separate thread."""
    global current_frame, detection_active, detection_stats, defect_log_cooldown
    
    logger.info("Detection loop started")
    
    while detection_active:
        try:
            # Read frame from camera
            result = camera.read_frame()
            if result is None:
                time.sleep(0.1)
                continue
            
            success, frame = result
            if not success or frame is None:
                time.sleep(0.1)
                continue
            
            # Perform detection
            detections = detector.detect(frame)
            
            # Draw detections on frame
            annotated_frame = detector.draw_detections(frame, detections)
            
            # Update current frame
            with frame_lock:
                current_frame = annotated_frame.copy()
            
            # Update stats
            detection_stats["current_defects"] = len(detections)
            if detections:
                detection_stats["total_detected"] += len(detections)
            
            # Log defects to database (with cooldown)
            current_time = time.time()
            for det in detections:
                defect_type = det["class_name"]
                confidence = det["confidence"]
                bbox = det["bbox"]
                
                # Check cooldown
                last_log_time = defect_log_cooldown.get(defect_type, 0)
                if current_time - last_log_time >= config.DEFECT_LOGGING_COOLDOWN:
                    # Log defect
                    if db and db.is_connected:
                        db.log_defect(
                            defect_type=defect_type,
                            confidence=confidence,
                            frame=frame,  # Use original frame, not annotated
                            bbox=bbox,
                            timestamp=datetime.now()
                        )
                        defect_log_cooldown[defect_type] = current_time
            
            # Control frame rate
            time.sleep(1.0 / config.STREAM_FPS)
            
        except Exception as e:
            logger.error(f"Error in detection loop: {str(e)}")
            time.sleep(0.1)
    
    logger.info("Detection loop stopped")


def generate_frames():
    """Generator function for video streaming (MJPEG)."""
    global current_frame
    
    while True:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                # Send placeholder frame if no frame available
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "Waiting for camera...",
                    (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(1.0 / config.STREAM_FPS)


@app.route('/')
def index():
    """Home page - Live detection view."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Analysis dashboard page."""
    return render_template('dashboard.html')


@app.route('/history')
def history():
    """Defect history page."""
    return render_template('history.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route (MJPEG)."""
    if camera is None or not camera.is_initialized:
        # Return a placeholder frame if camera is not initialized
        def generate_placeholder():
            while True:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "Camera not initialized",
                    (120, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    "Check camera connection",
                    (80, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2
                )
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1)
        return Response(
            generate_placeholder(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/detection/start', methods=['POST'])
def start_detection():
    """Start defect detection."""
    global detection_active, detection_thread
    
    if detection_active:
        return jsonify({"status": "already_running", "message": "Detection already active"}), 200
    
    if camera is None or not camera.is_initialized:
        return jsonify({"status": "error", "message": "Camera not initialized"}), 500
    
    if detector is None:
        return jsonify({"status": "error", "message": "Detector not initialized"}), 500
    
    detection_active = True
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    
    logger.info("Detection started")
    return jsonify({"status": "started", "message": "Detection started successfully"})


@app.route('/api/detection/stop', methods=['POST'])
def stop_detection():
    """Stop defect detection."""
    global detection_active
    
    detection_active = False
    
    logger.info("Detection stopped")
    return jsonify({"status": "stopped", "message": "Detection stopped successfully"})


@app.route('/api/detection/status', methods=['GET'])
def get_detection_status():
    """Get current detection status and statistics."""
    global detection_stats
    
    return jsonify({
        "active": detection_active,
        "stats": detection_stats,
        "camera_initialized": camera.is_initialized if camera else False,
        "detector_loaded": detector.model is not None if detector else False,
        "database_connected": db.is_connected if db else False
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get defect detection statistics from database."""
    if db and db.is_connected:
        stats = db.get_statistics()
        time_series = db.get_time_series_data(hours=24)
        return jsonify({
            "statistics": stats,
            "time_series": time_series
        })
    else:
        return jsonify({
            "statistics": {
                "total_defects": 0,
                "total_bottles": 0,
                "defects_by_type": {},
                "recent_defects": 0
            },
            "time_series": []
        })


@app.route('/api/defects', methods=['GET'])
def get_defects():
    """Get defect history."""
    limit = request.args.get('limit', 100, type=int)
    skip = request.args.get('skip', 0, type=int)
    defect_type = request.args.get('type', None)
    
    if db and db.is_connected:
        defects = db.get_all_defects(limit=limit, skip=skip, defect_type=defect_type)
        return jsonify({"defects": defects})
    else:
        return jsonify({"defects": []})


@app.route('/api/camera/list', methods=['GET'])
def list_cameras():
    """List available cameras."""
    from camera import list_available_cameras
    cameras = list_available_cameras(max_index=10)
    return jsonify({"cameras": cameras})


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


def cleanup():
    """Cleanup resources on shutdown."""
    global detection_active, camera, db
    
    logger.info("Shutting down...")
    detection_active = False
    
    if camera:
        camera.release()
    
    if db:
        db.disconnect()


if __name__ == '__main__':
    # Set Flask to handle requests properly
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    
    # Initialize system (non-blocking - Flask will start regardless)
    try:
        init_system()
    except Exception as e:
        logger.error(f"Error during system initialization: {str(e)}")
        logger.warning("Flask server will start but some features may be unavailable.")
    
    try:
        # Run Flask app
        logger.info(f"Starting Flask server at http://{config.FLASK_HOST}:{config.FLASK_PORT}")
        logger.info("Open your browser and navigate to the URL above")
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        cleanup()

"""
Camera module for handling external USB webcam input.
Provides camera detection, initialization, and frame capture functionality.
"""

import cv2
import logging
from typing import Optional, Tuple, List
import config

logger = logging.getLogger(__name__)


class Camera:
    """
    Camera handler for external USB webcam.
    Provides methods to detect available cameras and capture frames.
    """
    
    def __init__(self, camera_index: int = None):
        """
        Initialize camera handler.
        
        Args:
            camera_index: Camera index to use. If None, uses config.CAMERA_INDEX
        """
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
        self.available_cameras = []
        
    def test_cameras(self, max_index: int = 5) -> List[int]:
        """
        Test and return list of available camera indices.
        
        Args:
            max_index: Maximum camera index to test (default: 5)
            
        Returns:
            List of available camera indices
        """
        available = []
        logger.info("Scanning for available cameras...")
        
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available.append(i)
                    logger.info(f"Camera found at index {i}")
                cap.release()
            else:
                logger.debug(f"Camera index {i} not available")
        
        self.available_cameras = available
        return available
    
    def initialize(self) -> bool:
        """
        Initialize the camera with configured settings.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            logger.warning("Camera already initialized")
            return True
        
        # Test cameras if configured (skipped for faster startup)
        if config.TEST_CAMERAS_ON_STARTUP:
            available = self.test_cameras(max_index=3)  # Reduced from 5 to 3 for faster scanning
            if not available:
                logger.error("No cameras detected!")
                return False
            
            if self.camera_index not in available:
                logger.warning(
                    f"Configured camera index {self.camera_index} not available. "
                    f"Available cameras: {available}"
                )
                if available:
                    logger.info(f"Using first available camera: {available[0]}")
                    self.camera_index = available[0]
                else:
                    return False
        
        # Initialize camera (fast path - direct initialization)
        logger.info(f"Initializing camera at index {self.camera_index}")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera at index {self.camera_index}")
            return False
        
        # Set camera properties (non-blocking - don't wait for confirmation)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        except Exception:
            pass  # Silently continue if properties can't be set
        
        # Quick frame test (single read, no retries)
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            logger.error("Camera opened but cannot read frames. Camera may be in use by another application.")
            self.cap.release()
            return False
        
        # Skip property reading for faster initialization (properties will be correct when used)
        logger.info("Camera initialized successfully.")
        
        self.is_initialized = True
        return True
    
    def read_frame(self) -> Optional[Tuple[bool, any]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame) or None if camera not initialized
        """
        if not self.is_initialized or self.cap is None:
            logger.error("Camera not initialized")
            return None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame from camera")
            return (False, None)
        
        return (True, frame)
    
    def get_frame(self) -> Optional[any]:
        """
        Get a single frame from the camera.
        Convenience method that returns only the frame (or None).
        
        Returns:
            Frame as numpy array or None
        """
        result = self.read_frame()
        if result is None:
            return None
        success, frame = result
        return frame if success else None
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_initialized = False
            logger.info("Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release()


def list_available_cameras(max_index: int = 5) -> List[int]:
    """
    Utility function to list all available cameras.
    
    Args:
        max_index: Maximum camera index to test
        
    Returns:
        List of available camera indices
    """
    cam = Camera()
    return cam.test_cameras(max_index)


if __name__ == "__main__":
    # Test script to list available cameras
    logging.basicConfig(level=logging.INFO)
    print("=" * 50)
    print("Camera Detection Test")
    print("=" * 50)
    
    cameras = list_available_cameras(max_index=10)
    
    if cameras:
        print(f"\nFound {len(cameras)} available camera(s):")
        for idx in cameras:
            print(f"  - Camera index {idx}")
        print(f"\nTo use an external webcam, set CAMERA_INDEX = {cameras[-1]} in config.py")
    else:
        print("\nNo cameras found. Please connect a camera and try again.")

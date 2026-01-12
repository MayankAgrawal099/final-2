"""
Database module for MongoDB operations.
Handles defect logging, retrieval, and statistics.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional
import base64
import io
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import cv2
import numpy as np
import config

logger = logging.getLogger(__name__)


class Database:
    """
    MongoDB handler for defect logging and retrieval.
    """
    
    def __init__(self):
        """Initialize database connection."""
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        Connect to MongoDB database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to MongoDB at {config.MONGODB_URI}")
            self.client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database and collection
            self.db = self.client[config.DATABASE_NAME]
            self.collection = self.db[config.COLLECTION_NAME]
            
            # Create index on timestamp for faster queries
            self.collection.create_index("timestamp")
            self.collection.create_index("defect_type")
            
            self.is_connected = True
            logger.info("MongoDB connection established successfully")
            return True
            
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            logger.error("Please ensure MongoDB is running: mongod")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self.client is not None:
            self.client.close()
            self.is_connected = False
            logger.info("MongoDB connection closed")
    
    def encode_image(self, frame: np.ndarray, format: str = '.jpg') -> str:
        """
        Encode frame as Base64 string.
        
        Args:
            frame: Input frame as numpy array
            format: Image format ('.jpg' or '.png')
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Encode image
            if format == '.jpg':
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
            else:
                _, buffer = cv2.imencode('.png', frame)
            
            # Convert to Base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
            
        except Exception as e:
            logger.error(f"Image encoding error: {str(e)}")
            return ""
    
    def log_defect(
        self,
        defect_type: str,
        confidence: float,
        frame: np.ndarray,
        bbox: List[int],
        timestamp: datetime = None
    ) -> Optional[str]:
        """
        Log a detected defect to the database.
        
        Args:
            defect_type: Type of defect
            confidence: Detection confidence score
            frame: Frame image (will be encoded to Base64)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            timestamp: Timestamp of detection (defaults to now)
            
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.is_connected:
            logger.warning("Database not connected. Defect not logged.")
            return None
        
        try:
            # Prepare document
            if timestamp is None:
                timestamp = datetime.now()
            
            # Encode image
            image_base64 = self.encode_image(frame)
            if not image_base64:
                logger.error("Failed to encode image")
                return None
            
            document = {
                "defect_type": defect_type,
                "confidence": float(confidence),
                "timestamp": timestamp,
                "image": image_base64,
                "bbox": bbox,
                "image_format": "jpg"
            }
            
            # Insert document
            result = self.collection.insert_one(document)
            logger.debug(f"Defect logged: {defect_type} (ID: {result.inserted_id})")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to log defect: {str(e)}")
            return None
    
    def get_all_defects(
        self,
        limit: int = 100,
        skip: int = 0,
        defect_type: str = None
    ) -> List[Dict]:
        """
        Retrieve defect records from database.
        
        Args:
            limit: Maximum number of records to return
            skip: Number of records to skip (for pagination)
            defect_type: Filter by defect type (optional)
            
        Returns:
            List of defect documents
        """
        if not self.is_connected:
            logger.warning("Database not connected")
            return []
        
        try:
            query = {}
            if defect_type:
                query["defect_type"] = defect_type
            
            cursor = self.collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
            
            defects = []
            for doc in cursor:
                # Convert ObjectId to string
                doc["_id"] = str(doc["_id"])
                defects.append(doc)
            
            return defects
            
        except Exception as e:
            logger.error(f"Failed to retrieve defects: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get defect detection statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.is_connected:
            return {
                "total_defects": 0,
                "total_bottles": 0,
                "defects_by_type": {},
                "recent_defects": 0
            }
        
        try:
            # Total defects
            total_defects = self.collection.count_documents({})
            
            # Defects by type
            pipeline = [
                {
                    "$group": {
                        "_id": "$defect_type",
                        "count": {"$sum": 1}
                    }
                }
            ]
            defects_by_type = {}
            for result in self.collection.aggregate(pipeline):
                defect_type = result["_id"]
                display_name = config.DEFECT_CLASS_NAMES.get(
                    defect_type,
                    defect_type.replace("_", " ").title()
                )
                defects_by_type[display_name] = result["count"]
            
            # Recent defects (last 24 hours)
            from datetime import timedelta
            yesterday = datetime.now() - timedelta(days=1)
            recent_defects = self.collection.count_documents({
                "timestamp": {"$gte": yesterday}
            })
            
            # For demo: assume each defect is from a different bottle
            # In production, you'd track unique bottle IDs
            total_bottles = total_defects  # Simplified
            
            return {
                "total_defects": total_defects,
                "total_bottles": total_bottles,
                "defects_by_type": defects_by_type,
                "recent_defects": recent_defects
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {
                "total_defects": 0,
                "total_bottles": 0,
                "defects_by_type": {},
                "recent_defects": 0
            }
    
    def get_time_series_data(self, hours: int = 24) -> List[Dict]:
        """
        Get defect counts grouped by time intervals.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of time series data points
        """
        if not self.is_connected:
            return []
        
        try:
            from datetime import timedelta
            start_time = datetime.now() - timedelta(hours=hours)
            
            # Group by hour
            pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": start_time}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "year": {"$year": "$timestamp"},
                            "month": {"$month": "$timestamp"},
                            "day": {"$dayOfMonth": "$timestamp"},
                            "hour": {"$hour": "$timestamp"}
                        },
                        "count": {"$sum": 1}
                    }
                },
                {
                    "$sort": {"_id": 1}
                }
            ]
            
            time_series = []
            for result in self.collection.aggregate(pipeline):
                time_series.append({
                    "timestamp": f"{result['_id']['year']}-{result['_id']['month']:02d}-{result['_id']['day']:02d} {result['_id']['hour']:02d}:00",
                    "count": result["count"]
                })
            
            return time_series
            
        except Exception as e:
            logger.error(f"Failed to get time series data: {str(e)}")
            return []
    
    def clear_all_defects(self) -> bool:
        """
        Clear all defect records from database.
        Use with caution!
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            result = self.collection.delete_many({})
            logger.info(f"Cleared {result.deleted_count} defect records")
            return True
        except Exception as e:
            logger.error(f"Failed to clear defects: {str(e)}")
            return False


if __name__ == "__main__":
    # Test database connection
    logging.basicConfig(level=logging.INFO)
    db = Database()
    if db.connect():
        print("Database connection successful!")
        stats = db.get_statistics()
        print(f"Statistics: {stats}")
        db.disconnect()
    else:
        print("Database connection failed. Please ensure MongoDB is running.")

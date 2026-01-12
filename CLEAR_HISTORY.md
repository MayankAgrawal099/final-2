# How to Remove Existing Defect History

This guide explains how to clear all defect history from the VisionIQ system.

## Method 1: Using Python Script (Recommended)

Create and run a simple Python script:

```python
# clear_history.py
from database import Database

db = Database()
if db.connect():
    count = db.collection.count_documents({})
    print(f"Found {count} defect records")
    
    response = input(f"Are you sure you want to delete all {count} records? (yes/no): ")
    if response.lower() == 'yes':
        result = db.clear_all_defects()
        if result:
            print("All defect history has been cleared successfully!")
        else:
            print("Error clearing defect history")
    else:
        print("Operation cancelled")
    db.disconnect()
else:
    print("Could not connect to database. Please ensure MongoDB is running.")
```

Run it:
```bash
python clear_history.py
```

## Method 2: Using MongoDB Shell (Direct)

Connect to MongoDB and delete all documents:

```bash
# Start MongoDB shell
mongo

# Switch to the database
use bottle_defect_detection

# View current count
db.defects.countDocuments()

# Delete all defect records
db.defects.deleteMany({})

# Verify deletion
db.defects.countDocuments()  # Should return 0

# Exit
exit
```

## Method 3: Using MongoDB Compass (GUI)

1. Open MongoDB Compass
2. Connect to `mongodb://localhost:27017/`
3. Navigate to `bottle_defect_detection` database
4. Click on `defects` collection
5. Click the "Delete" button or use the filter `{}` and delete all
6. Confirm deletion

## Method 4: Delete Entire Database (Nuclear Option)

⚠️ **Warning**: This deletes the entire database, not just defects!

```bash
# MongoDB shell
mongo

use bottle_defect_detection
db.dropDatabase()

exit
```

## Quick One-Liner (MongoDB Shell)

```bash
mongo bottle_defect_detection --eval "db.defects.deleteMany({})"
```

## Verification

After clearing, verify the history is empty:

1. **Via Web Interface**: Go to History page - should show "No defects found"
2. **Via MongoDB**: Run `db.defects.countDocuments()` - should return 0

## Notes

- Defect images are stored as Base64 strings in MongoDB, so deleting records also deletes all images
- There is no "undo" - deleted records cannot be recovered
- The database structure remains intact (only data is deleted)
- Statistics on the Dashboard will reset to zero after clearing

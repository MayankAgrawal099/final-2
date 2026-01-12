# Complete Guide: Remove All Defect Data from VisionIQ

This guide explains how to completely remove all defect metadata, captured defects, analysis visuals, history, and statistics from the VisionIQ system.

## ⚠️ WARNING
**This action is PERMANENT and CANNOT be undone!** All defect images, statistics, and history will be permanently deleted.

---

## Method 1: Using Python Script (Easiest)

### Step 1: Run the Clear History Script

```bash
python clear_history.py
```

This will:
- Connect to MongoDB
- Show you how many records exist
- Ask for confirmation
- Delete all defect records
- Clear all statistics

### Step 2: Verify Deletion

After running the script, verify:
1. **Web Interface**: Go to History page - should show "No defects found"
2. **Dashboard**: All statistics should show 0
3. **MongoDB**: Run verification command (see Method 2)

---

## Method 2: Using MongoDB Shell (Direct)

### Step 1: Connect to MongoDB

```bash
mongo
```

### Step 2: Switch to Database

```javascript
use bottle_defect_detection
```

### Step 3: View Current Data

```javascript
// Count total defects
db.defects.countDocuments()

// View sample records
db.defects.find().limit(5).pretty()
```

### Step 4: Delete All Defect Records

```javascript
// Delete all defects
db.defects.deleteMany({})
```

### Step 5: Verify Deletion

```javascript
// Should return 0
db.defects.countDocuments()

// Should return empty array
db.defects.find()
```

### Step 6: Exit MongoDB

```javascript
exit
```

---

## Method 3: One-Line Command (Fastest)

### Windows PowerShell:
```powershell
mongo bottle_defect_detection --eval "db.defects.deleteMany({})"
```

### Linux/macOS:
```bash
mongo bottle_defect_detection --eval "db.defects.deleteMany({})"
```

---

## Method 4: Delete Entire Database (Nuclear Option)

⚠️ **This deletes the entire database, not just defects!**

### Step 1: Connect to MongoDB

```bash
mongo
```

### Step 2: Drop Database

```javascript
use bottle_defect_detection
db.dropDatabase()
```

### Step 3: Verify

```javascript
show dbs
// bottle_defect_detection should not appear in the list
```

### Step 4: Exit

```javascript
exit
```

---

## Method 5: Using MongoDB Compass (GUI)

### Step 1: Open MongoDB Compass
- Connect to `mongodb://localhost:27017/`

### Step 2: Navigate to Database
- Click on `bottle_defect_detection` database
- Click on `defects` collection

### Step 3: Delete All Records
- Click the "Filter" button
- Enter filter: `{}` (empty filter = all records)
- Click "Find"
- Click the "Delete" button (trash icon)
- Select "Delete All" or "Delete Many"
- Confirm deletion

### Step 4: Verify
- Collection should show 0 documents
- History page should be empty

---

## What Gets Deleted

When you clear defect data, the following are removed:

✅ **Defect Images**: All Base64-encoded images stored in MongoDB  
✅ **Defect Metadata**: Timestamps, confidence scores, bounding boxes  
✅ **Defect History**: All records in the History page  
✅ **Statistics**: Dashboard statistics reset to zero:
   - Total bottles inspected
   - Total defects detected
   - Defect distribution charts
   - Time-based trends

---

## What Remains Intact

The following are NOT deleted:

✅ **Database Structure**: Collection and indexes remain  
✅ **System Configuration**: All config.py settings  
✅ **YOLO Model**: Model files remain unchanged  
✅ **Application Code**: All Python files remain  
✅ **Web Interface**: All pages remain functional  

---

## Verification Checklist

After clearing data, verify:

- [ ] History page shows "No defects found"
- [ ] Dashboard shows all zeros:
  - [ ] Total Bottles Inspected: 0
  - [ ] Total Defects Detected: 0
  - [ ] Recent Defects (24h): 0
- [ ] Defect distribution chart is empty
- [ ] Trends chart shows no data
- [ ] MongoDB collection is empty: `db.defects.countDocuments()` returns 0

---

## Troubleshooting

### Issue: "Database not connected"
**Solution**: Ensure MongoDB is running:
```bash
# Windows (if running as service, it should auto-start)
# Or manually:
mongod

# Linux
sudo systemctl start mongodb

# macOS
brew services start mongodb-community
```

### Issue: "Permission denied"
**Solution**: 
- Windows: Run PowerShell/CMD as Administrator
- Linux/macOS: Use `sudo` if needed

### Issue: "Collection still has data"
**Solution**: 
- Verify you're connected to the correct database
- Check database name: `db.getName()`
- Try dropping and recreating collection:
```javascript
db.defects.drop()
```

---

## Quick Reference Commands

```bash
# Count defects
mongo bottle_defect_detection --eval "db.defects.countDocuments()"

# Delete all defects
mongo bottle_defect_detection --eval "db.defects.deleteMany({})"

# View all defects (before deletion)
mongo bottle_defect_detection --eval "db.defects.find().pretty()"

# Delete and verify in one command
mongo bottle_defect_detection --eval "db.defects.deleteMany({}); print('Deleted:', db.defects.countDocuments(), 'remaining')"
```

---

## After Clearing Data

Once data is cleared:

1. **System is ready**: The system will continue working normally
2. **New detections**: New defects will be logged starting from zero
3. **Statistics reset**: All counters and charts start fresh
4. **No recovery**: Deleted data cannot be recovered

---

## Need Help?

If you encounter issues:
1. Check MongoDB is running
2. Verify database name matches `config.py` (default: `bottle_defect_detection`)
3. Check MongoDB connection string in `config.py`
4. Review MongoDB logs for errors

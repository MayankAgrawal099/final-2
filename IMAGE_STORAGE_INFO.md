# Defect Image Storage Information

## Where Images Are Stored

Defect images from the defect history are stored **directly in MongoDB database** as **Base64-encoded strings**, not as separate image files.

### Storage Details:

- **Database**: MongoDB
- **Database Name**: `bottle_defect_detection` (configured in `config.py`)
- **Collection Name**: `defects` (configured in `config.py`)
- **Storage Format**: Base64-encoded JPEG images
- **Storage Location**: Inside each defect document in the `image` field

### How It Works:

1. When a defect is detected, the frame (image) is captured
2. The image is encoded to JPEG format (85% quality)
3. The JPEG is converted to a Base64 string
4. The Base64 string is stored in the MongoDB document's `image` field
5. Each defect document also stores:
   - `defect_type`: Type of defect
   - `confidence`: Detection confidence score
   - `timestamp`: When the defect was detected
   - `bbox`: Bounding box coordinates
   - `image_format`: "jpg"
   - `image`: Base64-encoded image string

### Database Connection:

- **URI**: `mongodb://localhost:27017/` (default)
- To view/access: Use MongoDB Compass or `mongo` shell
- Database location: Default MongoDB data directory (usually `/data/db` on Linux/Mac, or MongoDB installation directory on Windows)

### Benefits of This Approach:

- ✅ All data (images + metadata) in one place
- ✅ No separate file management needed
- ✅ Easy backup/restore (just backup MongoDB)
- ✅ Atomic operations (image + metadata stored together)

### Accessing Images:

Images are retrieved from MongoDB and converted back to display format in the web interface. The history page displays them by:
1. Retrieving the Base64 string from MongoDB
2. Creating a data URI: `data:image/jpg;base64,{base64_string}`
3. Displaying in the browser using the data URI

### To View Raw Data:

```bash
# Connect to MongoDB
mongo

# Use the database
use bottle_defect_detection

# View defect documents
db.defects.find().pretty()

# Count total defects
db.defects.countDocuments()

# View one defect document (with image data)
db.defects.findOne()
```

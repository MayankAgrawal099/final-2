# Complete Guide: Training Custom YOLO Model for Real-Time Water Bottle Defect Detection

This comprehensive guide will walk you through creating a production-ready YOLO model for VisionIQ water bottle defect detection system.

> **💡 Limited Images?** See `FEW_SHOT_LEARNING_GUIDE.md` for training with 10-50 images per class using few-shot learning techniques!

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Collection](#dataset-collection)
3. [Dataset Annotation](#dataset-annotation)
4. [Dataset Organization](#dataset-organization)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Integration with VisionIQ](#integration-with-visioniq)
8. [Testing and Deployment](#testing-and-deployment)

---

## Overview

### What You'll Create

A custom YOLOv8 model trained specifically to detect:
1. **Crack** - Bottle cracks or fractures
2. **Scratch/Mark** - Surface scratches or marks
3. **Missing Label** - Bottle without label
4. **Wrong Label** - Incorrect label on bottle
5. **Missing Cap** - Bottle without cap
6. **Wrong Cap Color** - Cap color doesn't match specification

### Requirements

- Python 3.8+
- Ultralytics YOLOv8
- Image annotation tool (LabelImg, CVAT, or Roboflow)
- GPU recommended (but CPU works too)
- 500+ images minimum (1000+ recommended)

---

## Step 1: Dataset Collection

### 1.1 Image Collection

**Where to Collect Images:**

1. **Production Line Photos**
   - Take photos directly from production line
   - Capture bottles on conveyor belt
   - Use same camera/lighting as production

2. **Controlled Environment**
   - Set up test station with consistent lighting
   - Use same background/background removal
   - Multiple angles per bottle

3. **Variety is Key**
   - Different bottle sizes and shapes
   - Various lighting conditions
   - Different backgrounds
   - Multiple angles (front, side, top)
   - Both defective and non-defective bottles

**Image Requirements:**

- **Format**: JPG or PNG
- **Resolution**: Minimum 640x640 pixels (1280x720+ recommended)
- **Quantity**: 
  - Minimum: 100 images per defect type (600 total)
  - Recommended: 200+ images per defect type (1200+ total)
  - Ideal: 500+ images per defect type (3000+ total)

**Image Naming Convention:**

```
bottle_001.jpg
bottle_002.jpg
bottle_003.jpg
...
```

### 1.2 Image Categories

Create images showing:

**Crack:**
- Visible cracks on bottle body
- Cracks at neck/base
- Hairline cracks
- Major fractures

**Scratch/Mark:**
- Surface scratches
- Scuff marks
- Discoloration marks
- Manufacturing marks

**Missing Label:**
- Bottles without labels
- Partially missing labels
- Labels that fell off

**Wrong Label:**
- Incorrect product label
- Wrong brand label
- Misprinted labels
- Upside-down labels

**Missing Cap:**
- Open bottles
- Bottles without caps
- Partially capped bottles

**Wrong Cap Color:**
- Blue cap on bottle that should have red
- Color mismatches
- Wrong cap type

---

## Step 2: Dataset Annotation

### 2.1 Choose Annotation Tool

**Option A: LabelImg (Recommended for Beginners)**

**Installation:**
```bash
pip install labelImg
```

**Usage:**
```bash
labelImg
```

**Features:**
- Free and open-source
- YOLO format support
- Easy to use
- Windows/Mac/Linux

**Option B: Roboflow (Cloud-based)**

**Features:**
- Web-based (no installation)
- Automatic augmentation
- Team collaboration
- Free tier available
- Visit: https://roboflow.com

**Option C: CVAT (Advanced)**

**Features:**
- Professional annotation tool
- Video annotation support
- Team collaboration
- Self-hosted or cloud

### 2.2 Annotation Process

**Using LabelImg:**

1. **Open LabelImg**
   ```bash
   labelImg
   ```

2. **Set Format**
   - Click "Change Save Dir"
   - Select YOLO format (not PascalVOC)
   - Format shown in bottom-right corner

3. **Load Images**
   - Click "Open Dir"
   - Select folder with your images

4. **Create Classes**
   - First image: Create class names
   - Classes: `crack`, `scratch`, `missing_label`, `wrong_label`, `missing_cap`, `wrong_cap_color`

5. **Draw Bounding Boxes**
   - Press `W` to create bounding box
   - Draw box around defect
   - Select class from dropdown
   - Save (Ctrl+S)

6. **Navigate**
   - `D` = Next image
   - `A` = Previous image
   - `W` = Create bounding box

**Annotation Guidelines:**

- **Box Tightness**: Draw boxes tightly around defects
- **Multiple Defects**: Annotate each defect separately
- **Partial Visibility**: Include partially visible defects
- **Occlusion**: Annotate even if partially hidden
- **Consistency**: Use same annotation style throughout

**Example Annotation:**

```
Image: bottle_001.jpg
- Box 1: [x1=120, y1=200, x2=180, y2=250] - Class: crack
- Box 2: [x1=300, y1=150, x2=350, y2=200] - Class: scratch
```

---

## Step 3: Dataset Organization

### 3.1 Folder Structure

Create the following folder structure:

```
bottle_defect_dataset/
├── images/
│   ├── train/
│   │   ├── bottle_001.jpg
│   │   ├── bottle_002.jpg
│   │   ├── bottle_003.jpg
│   │   └── ... (70% of images)
│   ├── val/
│   │   ├── bottle_201.jpg
│   │   ├── bottle_202.jpg
│   │   ├── bottle_203.jpg
│   │   └── ... (20% of images)
│   └── test/
│       ├── bottle_401.jpg
│       ├── bottle_402.jpg
│       ├── bottle_403.jpg
│       └── ... (10% of images)
├── labels/
│   ├── train/
│   │   ├── bottle_001.txt
│   │   ├── bottle_002.txt
│   │   ├── bottle_003.txt
│   │   └── ... (matching train images)
│   ├── val/
│   │   ├── bottle_201.txt
│   │   ├── bottle_202.txt
│   │   ├── bottle_203.txt
│   │   └── ... (matching val images)
│   └── test/
│       ├── bottle_401.txt
│       ├── bottle_402.txt
│       ├── bottle_403.txt
│       └── ... (matching test images)
└── data.yaml
```

### 3.2 Split Dataset

**Recommended Split:**
- **Training**: 70% of images
- **Validation**: 20% of images
- **Test**: 10% of images

**Python Script to Split Dataset:**

```python
# split_dataset.py
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_images, source_labels, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split dataset into train/val/test sets
    """
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f'images/{split}', exist_ok=True)
        os.makedirs(f'labels/{split}', exist_ok=True)
    
    # Get all image files
    images = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.png'))]
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split and copy
    for i, img in enumerate(images):
        label = img.replace('.jpg', '.txt').replace('.png', '.txt')
        
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'
        
        # Copy image
        shutil.copy(f'{source_images}/{img}', f'images/{split}/{img}')
        
        # Copy label if exists
        if os.path.exists(f'{source_labels}/{label}'):
            shutil.copy(f'{source_labels}/{label}', f'labels/{split}/{label}')

# Usage
split_dataset('raw_images', 'raw_labels')
```

### 3.3 Label File Format (YOLO)

Each `.txt` file contains one line per defect:

**Format:**
```
class_id center_x center_y width height
```

**Example: `bottle_001.txt`:**
```
0 0.5 0.3 0.1 0.15
1 0.7 0.6 0.08 0.12
```

**Coordinates are normalized (0-1):**
- `center_x`: X coordinate of box center / image width
- `center_y`: Y coordinate of box center / image height
- `width`: Box width / image width
- `height`: Box height / image height

**Class IDs:**
- `0` = crack
- `1` = scratch
- `2` = missing_label
- `3` = wrong_label
- `4` = missing_cap
- `5` = wrong_cap_color

### 3.4 Create data.yaml

Create `data.yaml` file:

```yaml
# data.yaml
path: /path/to/bottle_defect_dataset  # Dataset root directory
train: images/train  # Training images (relative to 'path')
val: images/val      # Validation images (relative to 'path')
test: images/test    # Test images (optional)

# Number of classes
nc: 6

# Class names (MUST match config.py DEFECT_CLASSES order)
names:
  0: crack
  1: scratch
  2: missing_label
  3: wrong_label
  4: missing_cap
  5: wrong_cap_color
```

**Important:** Class order MUST match `config.py` DEFECT_CLASSES!

---

## Step 4: Model Training

### 4.1 Install Dependencies

```bash
pip install ultralytics
pip install torch torchvision  # GPU version if available
```

### 4.2 Training Script

Create `train_model.py`:

```python
# train_model.py
from ultralytics import YOLO
import os

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # nano - fastest
# model = YOLO('yolov8s.pt')  # small - balanced
# model = YOLO('yolov8m.pt')  # medium - better accuracy
# model = YOLO('yolov8l.pt')  # large - high accuracy
# model = YOLO('yolov8x.pt')  # xlarge - best accuracy

# Train the model
results = model.train(
    data='data.yaml',           # Path to data.yaml
    epochs=100,                 # Number of training epochs
    imgsz=640,                  # Image size (640x640)
    batch=16,                   # Batch size (adjust based on GPU memory)
    name='bottle_defects',      # Project name
    device=0,                   # GPU device (0, 1, etc.) or 'cpu'
    patience=50,                # Early stopping patience
    save=True,                  # Save checkpoints
    save_period=10,             # Save checkpoint every N epochs
    val=True,                   # Validate during training
    plots=True,                 # Generate plots
    verbose=True,               # Verbose output
    # Augmentation (optional)
    hsv_h=0.015,               # HSV-Hue augmentation
    hsv_s=0.7,                 # HSV-Saturation augmentation
    hsv_v=0.4,                 # HSV-Value augmentation
    degrees=0.0,               # Rotation augmentation
    translate=0.1,              # Translation augmentation
    scale=0.5,                  # Scale augmentation
    flipud=0.0,                 # Vertical flip
    fliplr=0.5,                 # Horizontal flip
    mosaic=1.0,                 # Mosaic augmentation
    mixup=0.0,                  # Mixup augmentation
)

print("Training completed!")
print(f"Best model saved at: runs/detect/bottle_defects/weights/best.pt")
```

### 4.3 Run Training

```bash
python train_model.py
```

**Training Time:**
- CPU: 10-20 hours (depending on dataset size)
- GPU (NVIDIA): 1-3 hours
- GPU (CUDA): 2-5 hours

### 4.4 Monitor Training

Training outputs:
- **Loss curves**: `runs/detect/bottle_defects/results.png`
- **Validation metrics**: Console output
- **Best model**: `runs/detect/bottle_defects/weights/best.pt`
- **Last checkpoint**: `runs/detect/bottle_defects/weights/last.pt`

**Key Metrics to Watch:**
- **mAP50**: Mean Average Precision at IoU=0.5 (should be >0.7)
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95 (should be >0.5)
- **Precision**: Should be >0.8
- **Recall**: Should be >0.7

---

## Step 5: Model Evaluation

### 5.1 Evaluate on Test Set

```python
# evaluate_model.py
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/bottle_defects/weights/best.pt')

# Evaluate on test set
results = model.val(
    data='data.yaml',
    split='test',
    imgsz=640,
    conf=0.5,
    iou=0.45,
)

print(f"mAP50: {results.box.map50}")
print(f"mAP50-95: {results.box.map}")
print(f"Precision: {results.box.p}")
print(f"Recall: {results.box.r}")
```

### 5.2 Test on Sample Images

```python
# test_model.py
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('runs/detect/bottle_defects/weights/best.pt')

# Test on image
results = model('test_image.jpg', conf=0.5)

# Visualize results
results[0].show()

# Save results
results[0].save('result.jpg')
```

---

## Step 6: Integration with VisionIQ

### 6.1 Copy Model to Project

```bash
# Copy best model to project directory
cp runs/detect/bottle_defects/weights/best.pt ./bottle_defects_model.pt
```

### 6.2 Update config.py

Edit `config.py`:

```python
# Change MODEL_PATH to your trained model
MODEL_PATH = "bottle_defects_model.pt"  # Path to your trained model

# Verify DEFECT_CLASSES matches your training classes
DEFECT_CLASSES = {
    0: "crack",
    1: "scratch",
    2: "missing_label",
    3: "wrong_label",
    4: "missing_cap",
    5: "wrong_cap_color"
}
```

### 6.3 Test Integration

```bash
# Run the application
python app.py

# Test detection
# Go to Live Detection page
# Click "Start Detection"
# Verify detections match your trained classes
```

---

## Step 7: Testing and Deployment

### 7.1 Performance Testing

**Test Scenarios:**

1. **Single Defect Detection**
   - Test each defect type individually
   - Verify correct bounding boxes
   - Check confidence scores

2. **Multiple Defects**
   - Test bottles with multiple defects
   - Verify all defects detected
   - Check no false positives

3. **Edge Cases**
   - Partially visible defects
   - Different lighting conditions
   - Various bottle angles
   - Different bottle sizes

4. **False Positive Testing**
   - Normal bottles (should detect nothing)
   - Similar objects (should not detect)
   - Background objects (should ignore)

### 7.2 Optimization

**If Model Performance is Poor:**

1. **More Data**
   - Collect more images
   - Focus on underrepresented classes
   - Add more variety

2. **Data Augmentation**
   - Increase augmentation in training
   - Add more diverse backgrounds
   - Vary lighting conditions

3. **Model Size**
   - Try larger model (yolov8s, yolov8m)
   - Increase image size (640→1280)
   - More training epochs

4. **Hyperparameter Tuning**
   - Adjust learning rate
   - Modify batch size
   - Change augmentation parameters

### 7.3 Deployment Checklist

- [ ] Model trained and evaluated
- [ ] Model copied to project directory
- [ ] config.py updated with model path
- [ ] DEFECT_CLASSES verified
- [ ] Tested on sample images
- [ ] Tested on live camera feed
- [ ] Performance meets requirements
- [ ] False positive rate acceptable
- [ ] System runs smoothly in production

---

## Troubleshooting

### Issue: Low mAP Score
**Solutions:**
- Collect more training data
- Improve annotation quality
- Increase model size
- Adjust augmentation

### Issue: High False Positive Rate
**Solutions:**
- Increase confidence threshold
- Add negative examples (normal bottles)
- Improve training data quality
- Fine-tune model

### Issue: Missing Defects
**Solutions:**
- Lower confidence threshold
- Add more training examples
- Check annotation quality
- Increase model size

### Issue: Slow Inference
**Solutions:**
- Use smaller model (yolov8n)
- Reduce image size
- Use GPU acceleration
- Optimize batch processing

---

## Additional Resources

- **Ultralytics Documentation**: https://docs.ultralytics.com
- **YOLOv8 GitHub**: https://github.com/ultralytics/ultralytics
- **LabelImg Tutorial**: https://github.com/HumanSignal/labelImg
- **Roboflow**: https://roboflow.com
- **YOLO Format Guide**: https://docs.ultralytics.com/datasets/

---

## Quick Reference

**Training Command:**
```bash
python train_model.py
```

**Model Location:**
```
runs/detect/bottle_defects/weights/best.pt
```

**Config Update:**
```python
MODEL_PATH = "bottle_defects_model.pt"
```

**Test Command:**
```bash
python app.py
```

---

## Summary

1. **Collect** 1000+ images of bottles with defects
2. **Annotate** using LabelImg or Roboflow
3. **Organize** into train/val/test folders
4. **Train** YOLOv8 model
5. **Evaluate** on test set
6. **Integrate** with VisionIQ
7. **Test** and deploy

Good luck with your training! 🚀

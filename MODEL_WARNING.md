# ⚠️ IMPORTANT: Custom Model Required

## Current Issue

The system is currently using a **pretrained YOLOv8 model** (`yolov8n.pt`) which is trained on the **COCO dataset**. 

### Why This Doesn't Work

- **COCO Model**: Detects 80 general object classes (person, car, bottle, cup, etc.)
- **Your System**: Needs to detect 6 specific bottle defect types (crack, scratch, missing_label, etc.)
- **Result**: The pretrained model detects people, hands, faces, and other objects, NOT bottle defects!

### What's Happening

When the COCO model detects:
- **Person** (COCO class 0) → System incorrectly maps to "crack" (your class 0)
- **Hand** (COCO class) → Gets filtered out or incorrectly mapped
- **Face** (COCO class) → Gets filtered out or incorrectly mapped

**This is why you're seeing incorrect detections!**

---

## Solution: Train a Custom Model

You **MUST** train a custom YOLO model specifically for bottle defect detection.

### Quick Start

1. **Read the Training Guide**: `TRAINING_GUIDE.md`
2. **Collect Dataset**: 1000+ images of bottles with defects
3. **Annotate Images**: Use LabelImg or Roboflow
4. **Train Model**: Follow the guide step-by-step
5. **Update Config**: Point `MODEL_PATH` to your trained model

### Minimum Requirements

- **Images**: 100+ per defect type (600+ total minimum)
- **Annotation Tool**: LabelImg (free) or Roboflow (cloud-based)
- **Training Time**: 1-3 hours (GPU) or 10-20 hours (CPU)
- **Result**: Custom model that detects ONLY your 6 defect types

---

## Temporary Fix (Until Model is Trained)

The system has been updated to:
- **Detect** when using pretrained COCO model
- **Disable** defect detection (returns empty results)
- **Show warning** in logs

This prevents incorrect detections, but you won't see any defects until you train a custom model.

---

## Next Steps

1. ✅ System now prevents incorrect detections
2. 📖 Read `TRAINING_GUIDE.md`
3. 📸 Collect bottle defect images
4. 🏷️ Annotate your dataset
5. 🚀 Train your custom model
6. ⚙️ Update `config.py` with your model path
7. ✅ System will work correctly!

---

## Need Help?

- **Training Guide**: See `TRAINING_GUIDE.md` for complete instructions
- **Dataset Collection**: Guide includes where and how to collect images
- **Annotation**: Step-by-step LabelImg tutorial included
- **Model Training**: Complete training script provided

**The pretrained model is for DEMONSTRATION ONLY. For production use, you MUST train a custom model!**

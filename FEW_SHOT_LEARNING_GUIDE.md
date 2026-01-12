# Few-Shot Learning Guide: Training with Limited Images

This guide explains how to train VisionIQ with a limited number of images using few-shot learning techniques.

---

## ⚠️ Reality Check

**Traditional YOLO Training:**
- **Recommended**: 200+ images per class (1200+ total)
- **Minimum**: 100 images per class (600+ total)
- **Few-Shot**: 10-50 images per class (60-300 total)

**Few-shot learning is challenging** but possible with the right techniques!

---

## What is Few-Shot Learning?

Few-shot learning aims to train models with very few examples (typically 1-50 per class). For object detection, this is more difficult than classification because you need:
- Object localization (where is it?)
- Object classification (what is it?)

---

## Techniques for Few-Shot Training

### 1. Heavy Data Augmentation (Most Important!)

**This is your best friend with limited data!**

Augmentation creates variations of your images, effectively multiplying your dataset.

#### Augmentation Strategy:

```python
# train_model_fewshot.py
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=300,  # More epochs for few-shot
    imgsz=640,
    batch=8,  # Smaller batch for limited data
    
    # HEAVY AUGMENTATION - Critical for few-shot!
    hsv_h=0.02,        # Hue variation
    hsv_s=0.8,         # Saturation variation (high!)
    hsv_v=0.5,         # Value/brightness variation
    degrees=15.0,      # Rotation up to 15 degrees
    translate=0.2,     # Translation 20%
    scale=0.9,         # Scale variation
    shear=5.0,         # Shear transformation
    perspective=0.0005, # Perspective transformation
    flipud=0.5,        # Vertical flip 50%
    fliplr=0.5,        # Horizontal flip 50%
    mosaic=1.0,         # Mosaic augmentation (combines 4 images)
    mixup=0.3,         # Mixup augmentation (blends 2 images)
    copy_paste=0.3,    # Copy-paste augmentation (YOLOv8 feature)
    erasing=0.4,       # Random erasing
    crop_fraction=0.8, # Random cropping
)
```

**Why This Works:**
- **Mosaic**: Combines 4 images → 4x data
- **Mixup**: Blends 2 images → 2x data
- **Copy-Paste**: Copies objects between images → More examples
- **Rotation/Flip**: Creates 8+ variations per image

**Result**: 10 images can become 100+ training examples!

---

### 2. Transfer Learning (Use Pretrained Weights)

**Start from pretrained YOLOv8, not from scratch!**

```python
# Start from pretrained model (already done with yolov8n.pt)
model = YOLO('yolov8n.pt')  # ✅ Good - uses pretrained weights

# NOT this:
# model = YOLO('yolov8n.yaml')  # ❌ Bad - trains from scratch
```

**Why This Works:**
- Pretrained model already knows:
  - Object detection basics
  - Edge detection
  - Shape recognition
- You only need to fine-tune for your specific classes

---

### 3. Synthetic Data Generation

**Create artificial training data!**

#### Option A: Image Synthesis Tools

**Roboflow (Recommended):**
- Upload your few images
- Use "Generate" feature
- Creates synthetic variations automatically
- Visit: https://roboflow.com

**Albumentations:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.RandomBrightnessContrast(),
    A.ShiftScaleRotate(),
    A.GaussNoise(),
    A.Blur(),
    A.CLAHE(),
], bbox_params=A.BboxParams(format='yolo'))
```

#### Option B: 3D Rendering

- Create 3D models of bottles
- Render with defects
- Generate thousands of synthetic images
- Tools: Blender, Unity, Unreal Engine

#### Option C: GAN-Based Generation

- Use StyleGAN or similar
- Generate realistic bottle images
- Requires technical expertise

---

### 4. Active Learning

**Smart data collection strategy:**

1. **Start with few images** (10-20 per class)
2. **Train initial model**
3. **Test on real scenarios**
4. **Identify failure cases**
5. **Collect more images for those cases**
6. **Retrain**
7. **Repeat**

**Focus on:**
- Hard examples (where model fails)
- Edge cases (unusual angles, lighting)
- Rare defect types

---

### 5. Fine-Tuning Strategy

**Progressive fine-tuning:**

```python
# Stage 1: Freeze backbone, train only head
model = YOLO('yolov8n.pt')
model.train(
    data='data.yaml',
    epochs=50,
    freeze=10,  # Freeze first 10 layers
)

# Stage 2: Unfreeze, fine-tune all layers
model.train(
    data='data.yaml',
    epochs=100,
    resume=True,  # Continue from previous
)
```

---

## Practical Few-Shot Training Script

### Complete Training Script for Few-Shot Learning

```python
# train_fewshot.py
from ultralytics import YOLO
import os

print("=" * 60)
print("Few-Shot Learning Training for VisionIQ")
print("=" * 60)

# Check dataset size
def count_images(dataset_path):
    train_path = os.path.join(dataset_path, 'images', 'train')
    if os.path.exists(train_path):
        images = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.png'))]
        return len(images)
    return 0

dataset_size = count_images('.')
print(f"\nDataset size: {dataset_size} training images")

if dataset_size < 100:
    print("⚠️  Few-shot learning mode detected!")
    print("Using aggressive augmentation strategy...")
    
    # Few-shot configuration
    epochs = 300  # More epochs
    batch = 4     # Smaller batch
    patience = 100  # More patience for early stopping
    
    # Aggressive augmentation
    augmentation = {
        'hsv_h': 0.02,
        'hsv_s': 0.8,
        'hsv_v': 0.5,
        'degrees': 15.0,
        'translate': 0.2,
        'scale': 0.9,
        'shear': 5.0,
        'perspective': 0.0005,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.3,
        'copy_paste': 0.3,
    }
else:
    print("Standard training mode")
    epochs = 100
    batch = 16
    patience = 50
    augmentation = {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
    }

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train with few-shot optimized settings
results = model.train(
    data='data.yaml',
    epochs=epochs,
    imgsz=640,
    batch=batch,
    name='bottle_defects_fewshot',
    device=0,  # or 'cpu'
    patience=patience,
    save=True,
    save_period=10,
    val=True,
    plots=True,
    verbose=True,
    **augmentation  # Apply augmentation settings
)

print("\n" + "=" * 60)
print("Training completed!")
print(f"Best model: runs/detect/bottle_defects_fewshot/weights/best.pt")
print("=" * 60)
```

---

## Minimum Image Requirements

### Absolute Minimum (Not Recommended)

- **Per Class**: 5-10 images
- **Total**: 30-60 images
- **Expected Performance**: 30-50% mAP (poor)
- **Use Case**: Proof of concept only

### Practical Minimum

- **Per Class**: 20-30 images
- **Total**: 120-180 images
- **Expected Performance**: 50-70% mAP (acceptable)
- **Use Case**: Development/testing

### Recommended Minimum

- **Per Class**: 50-100 images
- **Total**: 300-600 images
- **Expected Performance**: 70-85% mAP (good)
- **Use Case**: Production (with augmentation)

---

## Data Collection Strategy for Few-Shot

### Maximize Diversity with Few Images

**For each defect type, collect:**

1. **Different Angles** (5 images)
   - Front view
   - Side view
   - Top view
   - 45° angle
   - Back view

2. **Different Lighting** (3 images)
   - Bright lighting
   - Normal lighting
   - Dim lighting

3. **Different Bottles** (5 images)
   - Different sizes
   - Different brands
   - Different colors

4. **Different Defect Severity** (3 images)
   - Minor defect
   - Moderate defect
   - Severe defect

**Total**: ~16 images per class × 6 classes = ~96 images

**With augmentation**: 96 images → 1000+ training examples!

---

## Augmentation Pipeline Example

```python
# advanced_augmentation.py
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Ultra-aggressive augmentation for few-shot
results = model.train(
    data='data.yaml',
    epochs=500,  # Many epochs
    
    # Geometric augmentations
    degrees=20.0,      # Rotate up to 20°
    translate=0.3,     # Move 30%
    scale=0.8,         # Scale 80-120%
    shear=10.0,        # Shear 10°
    perspective=0.001, # Perspective warp
    
    # Color augmentations
    hsv_h=0.03,        # Hue ±3%
    hsv_s=0.9,         # Saturation ±90%
    hsv_v=0.6,         # Brightness ±60%
    
    # Advanced augmentations
    mosaic=1.0,         # Always use mosaic
    mixup=0.5,         # 50% mixup
    copy_paste=0.5,    # 50% copy-paste
    erasing=0.5,       # Random erasing
    
    # Training settings
    batch=4,
    patience=150,
    lr0=0.01,          # Learning rate
    lrf=0.1,           # Final learning rate
    momentum=0.937,
    weight_decay=0.0005,
)
```

---

## Evaluation with Few-Shot

### Realistic Expectations

**With 10 images per class:**
- mAP50: 40-60% (acceptable for demo)
- Precision: 50-70%
- Recall: 40-60%

**With 50 images per class:**
- mAP50: 65-80% (good for production)
- Precision: 70-85%
- Recall: 60-75%

**With 100+ images per class:**
- mAP50: 80-90% (excellent)
- Precision: 85-95%
- Recall: 75-90%

---

## Tips for Success

### 1. Quality Over Quantity
- **Better**: 20 well-annotated, diverse images
- **Worse**: 100 poorly annotated, similar images

### 2. Focus on Hard Cases
- Collect images where defects are:
  - Partially visible
  - In different lighting
  - At unusual angles
  - On different bottle types

### 3. Use Validation Set
- Keep 20% for validation
- Don't train on validation data
- Monitor for overfitting

### 4. Early Stopping
- Use patience parameter
- Stop when validation loss stops improving
- Prevents overfitting with small datasets

### 5. Test Frequently
- Test on real production scenarios
- Identify failure cases
- Collect more data for those cases

---

## Limitations

### What Few-Shot Learning CAN'T Do Well

1. **Rare Defects**: If a defect appears in <5 images, model won't learn it
2. **Complex Scenes**: Multiple overlapping defects
3. **Novel Environments**: Very different from training data
4. **High Precision Requirements**: Medical/automotive applications

### When to Use Few-Shot

✅ **Good For:**
- Proof of concept
- Rapid prototyping
- Limited data availability
- Development/testing

❌ **Not Good For:**
- Production systems requiring high accuracy
- Safety-critical applications
- Regulated industries
- High-precision requirements

---

## Quick Start: Few-Shot Training

### Step 1: Collect Minimum Dataset

```
images/
├── train/
│   ├── crack_001.jpg (10 images)
│   ├── scratch_001.jpg (10 images)
│   ├── missing_label_001.jpg (10 images)
│   └── ... (6 classes × 10 images = 60 total)
└── val/
    └── ... (20% = 12 images)
```

### Step 2: Annotate

Use LabelImg or Roboflow to annotate all images.

### Step 3: Train with Augmentation

```bash
python train_fewshot.py
```

### Step 4: Evaluate

Check mAP score - aim for >50% with few-shot.

### Step 5: Iterate

- Test on real scenarios
- Collect more images for failures
- Retrain

---

## Conclusion

**Yes, you can train with few images using few-shot learning techniques!**

**Key Success Factors:**
1. ✅ Heavy data augmentation (most important!)
2. ✅ Transfer learning (use pretrained weights)
3. ✅ Diverse, high-quality images
4. ✅ More training epochs
5. ✅ Progressive fine-tuning

**Expected Results:**
- 10-20 images/class: 40-60% mAP (demo quality)
- 50+ images/class: 65-80% mAP (production quality)

**Remember**: Few-shot learning is a trade-off. More images = better results, but augmentation can help significantly!

---

## Resources

- **Roboflow**: https://roboflow.com (synthetic data generation)
- **Albumentations**: https://albumentations.ai (advanced augmentation)
- **YOLOv8 Docs**: https://docs.ultralytics.com
- **Few-Shot Learning Papers**: Search arXiv for "few-shot object detection"

Good luck with your few-shot training! 🚀

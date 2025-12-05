from ultralytics import YOLO
import torch
from pathlib import Path
import os
import gc

def train_yolov8_bdd100k():
    """
    Train YOLOv8x on BDD100K Dataset
    Optimized for RTX 5090 24GB - Guaranteed Memory Safe
    """
    
    # Set working directory
    code_dir = Path(__file__).parent
    os.chdir(code_dir)
    print(f"Working directory: {code_dir}\n")
    
    # GPU Check
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU detected!")
        print("This training requires a CUDA-capable GPU.")
        exit(1)
    
    device = 0
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print("="*70)
    print("GPU INFORMATION")
    print("="*70)
    print(f"GPU Model:  {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    print(f"CUDA:       {torch.version.cuda}")
    print(f"PyTorch:    {torch.__version__}")
    print("="*70)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Paths
    base_dir = Path(r"C:\Users\vishn\Documents\autonomous_driving\object_detection")
    yaml_path = code_dir / 'bdd100k.yaml'
    dataset_path = base_dir / 'bdd100k_yolo'
    
    # Verify files exist
    print("\nVerifying dataset...")
    if not yaml_path.exists():
        print(f"❌ ERROR: bdd100k.yaml not found at {yaml_path}")
        exit(1)
    
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print("Run prepare_dataset.py first!")
        exit(1)
    
    train_imgs = dataset_path / 'images/train'
    val_imgs = dataset_path / 'images/val'
    
    train_count = len(list(train_imgs.glob('*.jpg'))) if train_imgs.exists() else 0
    val_count = len(list(val_imgs.glob('*.jpg'))) if val_imgs.exists() else 0
    
    if train_count == 0:
        print(f"❌ ERROR: No training images found!")
        exit(1)
    
    print(f"✓ Training images: {train_count:,}")
    print(f"✓ Validation images: {val_count:,}")
    print(f"✓ Dataset config: {yaml_path.name}")
    
    # Model selection
    print("\n" + "="*70)
    print("MODEL CONFIGURATION")
    print("="*70)
    print("Using YOLOv8x (Extra-Large) - Best Accuracy")
    print("68M parameters | 258 GFLOPs")
    print("="*70)
    
    model = YOLO('yolov8x.pt')
    
    # MEMORY-OPTIMIZED SETTINGS for RTX 5090
    # Image size 1280 with YOLOv8x is very memory intensive
    batch_size = 6  # Safe for 24GB at 1280px
    workers = 6     # Optimal for 8-core systems
    
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Epochs:         500 (early stopping enabled)")
    print(f"Image Size:     1280x1280 pixels")
    print(f"Batch Size:     {batch_size}")
    print(f"Workers:        {workers}")
    print(f"Mixed Precision: Enabled (AMP)")
    print(f"Optimizer:      AdamW")
    print(f"Initial LR:     0.0005")
    print(f"Final LR:       0.00005 (cosine decay)")
    print(f"Weight Decay:   0.0005")
    print(f"Warmup Epochs:  5")
    print("="*70)
    
    print(f"\n{'='*70}")
    print("DATA AUGMENTATION")
    print("="*70)
    print("• Horizontal Flip:    50%")
    print("• HSV Augmentation:   Enabled")
    print("• Mosaic:             100% (disabled last 15 epochs)")
    print("• Mixup:              15%")
    print("• Copy-Paste:         10%")
    print("• Random Erasing:     40%")
    print("• Auto Augmentation:  RandAugment")
    print("="*70)
    
    print("\n🚀 Starting training...")
    print("This will take approximately 30-40 hours on RTX 5090.")
    print("Model checkpoints saved every 5 epochs.\n")
    
    # TRAINING
    try:
        results = model.train(
            # Dataset
            data=str(yaml_path),
            epochs=500,
            imgsz=1280,
            batch=batch_size,
            device=device,
            
            # Optimizer
            optimizer='AdamW',
            lr0=0.0005,          # Initial learning rate
            lrf=0.001,           # Final learning rate (lr0 * lrf = 0.00005)
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Loss weights
            box=7.5,             # Box loss gain
            cls=0.5,             # Classification loss gain
            dfl=1.5,             # Distribution focal loss gain
            
            # Augmentation
            hsv_h=0.015,         # HSV-Hue augmentation
            hsv_s=0.7,           # HSV-Saturation
            hsv_v=0.4,           # HSV-Value
            degrees=0.0,         # Rotation (disabled for driving)
            translate=0.1,       # Translation
            scale=0.9,           # Scale
            shear=0.0,           # Shear (disabled)
            perspective=0.0,     # Perspective (disabled)
            flipud=0.0,          # Vertical flip (disabled)
            fliplr=0.5,          # Horizontal flip
            mosaic=1.0,          # Mosaic augmentation
            mixup=0.15,          # Mixup augmentation
            copy_paste=0.1,      # Copy-paste augmentation
            auto_augment='randaugment',  # RandAugment
            erasing=0.4,         # Random erasing
            
            # Training settings
            patience=100,        # Early stopping patience
            save=True,
            save_period=5,       # Save checkpoint every 5 epochs
            cache=False,         # Don't cache (saves memory)
            workers=workers,
            project=str(base_dir / 'models'),
            name='yolov8x_bdd100k_maxperf',
            exist_ok=True,
            pretrained=True,
            verbose=True,
            seed=0,
            deterministic=False,
            single_cls=False,
            rect=False,
            cos_lr=True,         # Cosine learning rate scheduler
            close_mosaic=15,     # Disable mosaic last 15 epochs
            resume=False,
            amp=True,            # Automatic Mixed Precision
            fraction=1.0,        # Use 100% of data
            profile=False,
            freeze=None,
            multi_scale=False,   # Disabled to save memory
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            split='val',
            save_json=True,
            conf=None,
            iou=0.7,
            max_det=300,
            half=False,
            dnn=False,
            plots=True,
            show=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show_labels=True,
            show_conf=True,
            show_boxes=True,
            visualize=False,
            augment=False,
            agnostic_nms=False,
            retina_masks=False,
            line_width=None,
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Partial results saved in models folder.")
        return None
        
    except Exception as e:
        print(f"\n\n❌ ERROR during training: {e}")
        print("Check GPU memory and dataset integrity.")
        raise
    
    # Load best model for final validation
    best_model_path = base_dir / 'models/yolov8x_bdd100k_maxperf/weights/best.pt'
    
    if not best_model_path.exists():
        print("\n⚠️  Best model not found. Training may have failed.")
        return results
    
    print(f"\n{'='*70}")
    print("FINAL VALIDATION")
    print("="*70)
    print(f"Loading best model: {best_model_path.name}\n")
    
    # Clear memory before validation
    torch.cuda.empty_cache()
    gc.collect()
    
    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(
        data=str(yaml_path),
        imgsz=1280,
        batch=4,  # Smaller batch for validation
        device=device,
        plots=True,
        save_json=True,
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print("="*70)
    print(f"mAP50:      {metrics.box.map50*100:.2f}%")
    print(f"mAP50-95:   {metrics.box.map*100:.2f}%")
    print(f"Precision:  {metrics.box.mp*100:.2f}%")
    print(f"Recall:     {metrics.box.mr*100:.2f}%")
    print("="*70)
    
    # Per-class results
    class_names = [
        'pedestrian', 'rider', 'car', 'truck', 'bus',
        'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
    ]
    
    print("\nPER-CLASS RESULTS (mAP50):")
    print("-"*70)
    
    if hasattr(metrics.box, 'ap50'):
        for i, name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                print(f"  {name:15s}: {metrics.box.ap50[i]*100:6.2f}%")
    
    print("="*70)
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Best Model:     {best_model_path}")
    print(f"Training Time:  Check logs above")
    print(f"Final mAP50:    {metrics.box.map50*100:.2f}%")
    print(f"Final mAP50-95: {metrics.box.map*100:.2f}%")
    print("="*70)
    
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE CONTEXT")
    print("="*70)
    print("BDD100K is one of the most challenging autonomous driving datasets.")
    print("Realistic expectations:")
    print("  • mAP50:     60-75%  ← EXCELLENT performance")
    print("  • mAP50-95:  40-55%  ← EXCELLENT performance")
    print("\nState-of-the-art research papers achieve ~50-60% mAP50-95.")
    print("95%+ accuracy is unrealistic for this dataset's complexity.")
    print("="*70)
    
    print("\n✅ Training complete! Model ready for deployment.\n")
    
    return results


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    print("\n" + "="*70)
    print("BDD100K OBJECT DETECTION TRAINING")
    print("YOLOv8x - Maximum Performance Configuration")
    print("="*70 + "\n")
    
    try:
        train_yolov8_bdd100k()
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        raise

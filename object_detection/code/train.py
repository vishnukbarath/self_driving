from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

def train_yolov8_bdd100k():
    """Train YOLOv8 on BDD100K with maximum optimization for best accuracy"""
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠ WARNING: No GPU detected! Training will be extremely slow.")
        print("For best results, use a GPU (RTX 3090, 4090, or A100)")
        device = 'cpu'
    else:
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
    
    base_dir = Path(r"C:\Users\vishn\Documents\autonomous_driving\object_detection")
    
    # Use largest model for maximum accuracy
    print("\n" + "="*60)
    print("Using YOLOv8x (Extra-Large) for maximum accuracy")
    print("="*60)
    model = YOLO('yolov8x.pt')
    
    # Optimal batch size based on GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    if gpu_mem >= 24:
        batch = 16  # RTX 3090/4090 24GB
        print(f"✓ Batch size: 16 (optimal for {gpu_mem:.0f}GB GPU)")
    elif gpu_mem >= 12:
        batch = 8   # RTX 3080/3060 12GB
        print(f"✓ Batch size: 8 (optimal for {gpu_mem:.0f}GB GPU)")
    elif gpu_mem >= 8:
        batch = 4   # RTX 3070 8GB
        print(f"✓ Batch size: 4 (optimal for {gpu_mem:.0f}GB GPU)")
    else:
        batch = 2
        print(f"⚠ Batch size: 2 (limited GPU memory)")
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model:      YOLOv8x (Extra-Large)")
    print(f"Epochs:     500 (with early stopping)")
    print(f"Image size: 1280x1280")
    print(f"Batch size: {batch}")
    print(f"Device:     {device}")
    print("="*60)
    
    # Maximum accuracy training configuration
    results = model.train(
        # Data
        data='bdd100k.yaml',
        epochs=500,  # More epochs with early stopping
        imgsz=1280,  # Larger image size for better small object detection
        batch=batch,
        device=device,
        
        # Optimizer - AdamW best for complex datasets
        optimizer='AdamW',
        lr0=0.0005,  # Lower learning rate for stability
        lrf=0.001,   # Very low final LR for fine-tuning
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,  # Longer warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss function weights
        box=7.5,        # Box loss weight
        cls=0.5,        # Class loss weight
        dfl=1.5,        # Distribution focal loss weight
        
        # Augmentation - aggressive for robustness
        hsv_h=0.015,    # Hue augmentation
        hsv_s=0.7,      # Saturation
        hsv_v=0.4,      # Value/brightness
        degrees=0.0,    # No rotation (preserves horizon in driving)
        translate=0.1,  # Translation
        scale=0.9,      # Scaling (0.5 = 50% zoom in/out)
        shear=0.0,      # No shear (preserves geometry)
        perspective=0.0,  # No perspective (keeps lanes straight)
        flipud=0.0,     # No vertical flip (cars don't flip)
        fliplr=0.5,     # Horizontal flip (mirrors driving)
        mosaic=1.0,     # Mosaic augmentation
        mixup=0.15,     # Mixup augmentation (helps generalization)
        copy_paste=0.1, # Copy-paste augmentation
        auto_augment='randaugment',  # Additional augmentation
        erasing=0.4,    # Random erasing
        crop_fraction=1.0,  # Use full image
        
        # Training parameters
        patience=100,   # Early stopping patience (wait 100 epochs)
        save=True,
        save_period=5,  # Save checkpoint every 5 epochs
        cache=False,    # Set True if you have 64GB+ RAM (faster training)
        workers=8,      # Data loading workers
        project=str(base_dir / 'models'),
        name='yolov8x_bdd100k_maxperf',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=0,         # Fixed seed for reproducibility
        deterministic=False,  # Allow non-deterministic for speed
        single_cls=False,
        rect=False,     # Rectangular training (can help with varied aspect ratios)
        cos_lr=True,    # Cosine learning rate scheduler
        close_mosaic=15,  # Disable mosaic in last 15 epochs
        resume=False,
        amp=True,       # Automatic Mixed Precision (faster + less memory)
        fraction=1.0,   # Use 100% of dataset
        profile=False,
        freeze=None,    # Don't freeze any layers
        multi_scale=True,  # Multi-scale training (important!)
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,    # No dropout in YOLOv8
        val=True,
        split='val',
        save_json=True,  # Save results in COCO format
        save_hybrid=False,
        conf=None,
        iou=0.7,        # IoU threshold for NMS
        max_det=300,    # Max detections per image
        half=False,     # Use FP16 during validation
        dnn=False,
        plots=True,     # Generate training plots
        source=None,
        vid_stride=1,
        stream_buffer=False,
        visualize=False,
        augment=False,  # TTA (Test Time Augmentation) during val
        agnostic_nms=False,
        classes=None,
        retina_masks=False,
        embed=None,
        show=False,
        save_frames=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        line_width=None,
    )
    
    # Print final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Load best model and validate
    best_model_path = base_dir / 'models/yolov8x_bdd100k_maxperf/weights/best.pt'
    print(f"\n✓ Best model saved at:")
    print(f"  {best_model_path}")
    
    # Final validation
    print("\n" + "="*60)
    print("FINAL VALIDATION ON BEST MODEL")
    print("="*60)
    
    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(data='bdd100k.yaml', imgsz=1280, batch=batch)
    
    print("\n" + "="*60)
    print("FINAL METRICS")
    print("="*60)
    print(f"mAP50:      {metrics.box.map50*100:.2f}%")
    print(f"mAP50-95:   {metrics.box.map*100:.2f}%")
    print(f"Precision:  {metrics.box.mp*100:.2f}%")
    print(f"Recall:     {metrics.box.mr*100:.2f}%")
    print("="*60)
    
    # Per-class results
    class_names = ['pedestrian', 'rider', 'car', 'truck', 'bus', 
                   'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
    
    print("\nPer-Class mAP50:")
    for i, name in enumerate(class_names):
        if hasattr(metrics.box, 'ap_class_index') and i in metrics.box.ap_class_index:
            idx = list(metrics.box.ap_class_index).index(i)
            print(f"  {name:15s}: {metrics.box.ap50[idx]*100:.2f}%")
    
    # Expected results warning
    print("\n" + "="*60)
    print("EXPECTED PERFORMANCE ON BDD100K:")
    print("="*60)
    print("Realistic targets for this challenging dataset:")
    print("  • mAP50:     60-75% (EXCELLENT if achieved)")
    print("  • mAP50-95:  40-55% (EXCELLENT if achieved)")
    print("\nNote: 95%+ is unrealistic for BDD100K's complexity")
    print("State-of-the-art research models achieve ~50-60% mAP50-95")
    print("="*60)
    
    return results

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    train_yolov8_bdd100k()

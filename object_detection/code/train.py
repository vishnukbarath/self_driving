from ultralytics import YOLO
import torch
from pathlib import Path
import os

def train_yolov8_bdd100k():
    """Train YOLOv8 on BDD100K with maximum optimization for best accuracy"""
    
    # Set working directory to code folder
    code_dir = Path(__file__).parent
    os.chdir(code_dir)
    print(f"Working directory: {code_dir}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠ WARNING: No GPU detected! Training will be extremely slow.")
        device = 'cpu'
    else:
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
    
    base_dir = Path(r"C:\Users\vishn\Documents\autonomous_driving\object_detection")
    
    # Verify bdd100k.yaml exists
    yaml_path = code_dir / 'bdd100k.yaml'
    if not yaml_path.exists():
        print(f"✗ ERROR: bdd100k.yaml not found at {yaml_path}")
        print(f"\nPlease create bdd100k.yaml in: {code_dir}")
        exit(1)
    
    # Verify dataset exists
    dataset_path = base_dir / 'bdd100k_yolo'
    if not dataset_path.exists():
        print(f"✗ ERROR: Dataset not found at {dataset_path}")
        print(f"\nRun prepare_dataset.py first to create the dataset!")
        exit(1)
    
    train_imgs = dataset_path / 'images/train'
    if not train_imgs.exists() or len(list(train_imgs.glob('*.jpg'))) == 0:
        print(f"✗ ERROR: No training images found at {train_imgs}")
        print(f"\nRun prepare_dataset.py first!")
        exit(1)
    
    print(f"✓ Found {len(list(train_imgs.glob('*.jpg')))} training images")
    print(f"✓ Dataset config: {yaml_path}")
    
    # Use largest model for maximum accuracy
    print("\n" + "="*60)
    print("Using YOLOv8x (Extra-Large) for maximum accuracy")
    print("="*60)
    model = YOLO('yolov8x.pt')
    
    # Optimal batch size based on GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    if gpu_mem >= 24:
        batch = 16  # RTX 5090 24GB
        print(f"✓ Batch size: 16 (optimal for {gpu_mem:.0f}GB GPU)")
    elif gpu_mem >= 12:
        batch = 8
        print(f"✓ Batch size: 8 (optimal for {gpu_mem:.0f}GB GPU)")
    elif gpu_mem >= 8:
        batch = 4
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
        # Data - use absolute path
        data=str(yaml_path),
        epochs=500,
        imgsz=1280,
        batch=batch,
        device=device,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.0005,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.9,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        auto_augment='randaugment',
        erasing=0.4,
        
        # Training settings
        patience=100,
        save=True,
        save_period=5,
        cache=False,
        workers=8,
        project=str(base_dir / 'models'),
        name='yolov8x_bdd100k_maxperf',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=0,
        deterministic=False,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=15,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        multi_scale=True,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        split='val',
        save_json=True,
        save_hybrid=False,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        source=None,
        vid_stride=1,
        stream_buffer=False,
        visualize=False,
        augment=False,
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
    metrics = best_model.val(data=str(yaml_path), imgsz=1280, batch=batch)
    
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
    
    # Expected results
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

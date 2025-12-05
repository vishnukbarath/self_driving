import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# BDD100K class mapping (det_20)
BDD_CLASSES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus', 
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

def find_files(base_dir, pattern):
    """Recursively find files matching pattern"""
    return list(Path(base_dir).rglob(pattern))

def build_image_index(img_dir):
    """Build index of all images in directory and subdirectories"""
    print(f"Building image index from: {img_dir}")
    image_index = {}
    
    # Search for all jpg images recursively
    for img_path in img_dir.rglob("*.jpg"):
        image_index[img_path.name] = img_path
    
    print(f"✓ Indexed {len(image_index)} images")
    return image_index

def convert_bdd_to_yolo(json_path, img_dir, output_dir, split='train'):
    """Convert BDD100K detection JSON to YOLO format"""
    
    # Create output directories
    img_out = Path(output_dir) / 'images' / split
    lbl_out = Path(output_dir) / 'labels' / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    # Build image index first (searches all subdirectories)
    image_index = build_image_index(img_dir)
    
    # Load JSON annotations
    print(f"\nLoading annotations from: {json_path.name}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} images for {split} split...")
    
    processed = 0
    skipped = 0
    no_labels = 0
    
    for item in tqdm(data):
        img_name = item['name']
        
        # Look up image in index
        if img_name not in image_index:
            skipped += 1
            continue
        
        img_path = image_index[img_name]
            
        # Copy image to output directory
        shutil.copy(img_path, img_out / img_name)
        
        # Get image dimensions (BDD100K standard is 1280x720)
        img_width, img_height = 1280, 720
        
        # Create YOLO label file
        label_file = lbl_out / (Path(img_name).stem + '.txt')
        
        label_count = 0
        with open(label_file, 'w') as f:
            if 'labels' in item and item['labels']:
                for label in item['labels']:
                    # Skip if not a detection task or wrong category
                    if 'category' not in label or label['category'] not in BDD_CLASSES:
                        continue
                    
                    # Skip if no bounding box
                    if 'box2d' not in label:
                        continue
                    
                    class_id = BDD_CLASSES.index(label['category'])
                    
                    # Get bounding box
                    box = label['box2d']
                    x1, y1 = box['x1'], box['y1']
                    x2, y2 = box['x2'], box['y2']
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Clip to valid range [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # Write to file
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    label_count += 1
        
        if label_count == 0:
            no_labels += 1
        
        processed += 1
    
    print(f"\nCompleted {split} split!")
    print(f"  ✓ Processed: {processed} images")
    print(f"  ✓ Labels created: {processed - no_labels} images with objects")
    print(f"  ✗ Skipped: {skipped} images (not found)")
    print(f"  ⚠ Empty: {no_labels} images (no valid labels)")
    
    if skipped > 0:
        print(f"\n⚠ WARNING: {skipped} images from JSON were not found!")
        print(f"  Check if all image files are in: {img_dir}")

# Main conversion
if __name__ == "__main__":
    print("="*60)
    print("BDD100K to YOLO Format Converter - AUTO DETECT MODE")
    print("="*60)
    
    base_dir = Path(r"C:\Users\vishn\Documents\autonomous_driving")
    
    print(f"\nSearching for BDD100K files in: {base_dir}")
    print("This may take a moment...\n")
    
    # Auto-search for JSON label files
    print("Searching for label JSON files...")
    all_json_files = find_files(base_dir, "*.json")
    
    # Filter for detection labels (det, train, val keywords)
    train_candidates = [f for f in all_json_files if 'train' in f.stem.lower() and ('det' in f.stem.lower() or 'det' in str(f.parent))]
    val_candidates = [f for f in all_json_files if 'val' in f.stem.lower() and ('det' in f.stem.lower() or 'det' in str(f.parent))]
    
    # Select the first valid candidates
    train_json = train_candidates[0] if train_candidates else None
    val_json = val_candidates[0] if val_candidates else None
    
    if not train_json:
        any_train = [f for f in all_json_files if 'train' in f.stem.lower()]
        if any_train:
            train_json = any_train[0]
        else:
            print("No train JSON files found!")
            exit(1)
    
    if not val_json:
        any_val = [f for f in all_json_files if 'val' in f.stem.lower()]
        if any_val:
            val_json = any_val[0]
        else:
            print("No val JSON files found!")
            exit(1)
    
    print(f"✓ Selected train JSON: {train_json.relative_to(base_dir)}")
    print(f"✓ Selected val JSON: {val_json.relative_to(base_dir)}")
    
    # Find image directories
    train_imgs = base_dir / "data/bdd100k/bdd100k/images/100k/train"
    val_imgs = base_dir / "data/bdd100k/bdd100k/images/100k/val"
    
    if not train_imgs.exists():
        print(f"\nSearching for train images...")
        train_search = list(base_dir.rglob("100k/train"))
        if train_search:
            train_imgs = train_search[0]
        else:
            print("Train images not found!")
            exit(1)
    
    if not val_imgs.exists():
        print(f"\nSearching for val images...")
        val_search = list(base_dir.rglob("100k/val"))
        if val_search:
            val_imgs = val_search[0]
        else:
            print("Val images not found!")
            exit(1)
    
    print(f"✓ Train images: {train_imgs.relative_to(base_dir)}")
    print(f"✓ Val images: {val_imgs.relative_to(base_dir)}")
    
    # Output directory
    output_dir = base_dir / "object_detection/bdd100k_yolo"
    print(f"✓ Output directory: {output_dir.relative_to(base_dir)}")
    
    print("\n" + "="*60)
    print("Starting conversion...")
    print("="*60)
    
    # Convert train and val splits
    convert_bdd_to_yolo(train_json, train_imgs, output_dir, 'train')
    convert_bdd_to_yolo(val_json, val_imgs, output_dir, 'val')
    
    print("\n" + "="*60)
    print("✓ DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nYour YOLO dataset is ready at:")
    print(f"  {output_dir}")
    
    # Show dataset statistics
    train_img_count = len(list((output_dir / 'images' / 'train').glob('*.jpg')))
    val_img_count = len(list((output_dir / 'images' / 'val').glob('*.jpg')))
    
    print(f"\nDataset Statistics:")
    print(f"  Train: {train_img_count} images")
    print(f"  Val: {val_img_count} images")
    print(f"  Total: {train_img_count + val_img_count} images")
    
    if train_img_count > 50000:
        print(f"\n✓ Great! You have a full training set.")
    elif train_img_count > 1000:
        print(f"\n⚠ You have a partial training set. Training will still work.")
    else:
        print(f"\n⚠ WARNING: Very few training images. Check your data!")
    
    print(f"\nNext step: Run train.py to start training!")

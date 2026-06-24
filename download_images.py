import os
import shutil
import time
from PIL import Image
from better_bing_image_downloader.downloader import Downloader

DATASET_DIR = "Dataset"
TARGET_TOTAL_IMAGES = 80

# Refined keywords map for highly relevant results
KEYWORDS_MAP = {
    "Biodegradable": [
        "organic waste food",
        "compostable food waste",
        "vegetable fruit peel waste"
    ],
    "E-Waste": [
        "electronic waste scrap",
        "discarded computers keyboards mouse",
        "waste circuit boards computer parts"
    ],
    "Glass": [
        "glass bottles waste",
        "broken glass shards recycling",
        "discarded glass jars garbage"
    ],
    "Hazardous": [
        "hazardous waste paint cans",
        "used medical syringes waste",
        "expired medicine pills packaging waste",
        "toxic chemical bottles waste"
    ],
    "Metal": [
        "metal soda cans crushed",
        "scrap metal pieces waste",
        "tin cans garbage recycling"
    ],
    "Paper": [
        "waste paper sheets crumpled",
        "discarded cardboard boxes waste"
    ],
    "Plastic": [
        "plastic bottles waste",
        "single use plastic cups garbage",
        "plastic bags trash",
        "discarded plastic containers"
    ],
    "Textile": [
        "old clothes waste fabric",
        "discarded textile waste clothing",
        "fabric scraps rags trash"
    ]
}

def get_existing_count(class_dir):
    if not os.path.exists(class_dir):
        os.makedirs(class_dir, exist_ok=True)
        return 0
    # Count files that are valid images in the directory
    files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    max_idx = 0
    for f in files:
        name, ext = os.path.splitext(f)
        try:
            val = int(name)
            if val > max_idx:
                max_idx = val
        except ValueError:
            pass
    if max_idx == 0:
        return len(files)
    return max_idx

def main():
    print("Starting download script using better-bing-image-downloader (Downloader class)...")
    
    os.makedirs(DATASET_DIR, exist_ok=True)
    temp_dir = os.path.join(DATASET_DIR, "temp_downloads")
    
    dl = Downloader()
    
    for class_name, keywords in KEYWORDS_MAP.items():
        class_dir = os.path.join(DATASET_DIR, class_name)
        start_count = get_existing_count(class_dir)
        
        print(f"\n=========================================")
        print(f"Class: {class_name}")
        print(f"Current image count: {start_count}")
        
        if start_count >= TARGET_TOTAL_IMAGES:
            print(f"Class {class_name} already has {start_count} images (target is {TARGET_TOTAL_IMAGES}). Skipping.")
            continue
            
        needed = TARGET_TOTAL_IMAGES - start_count
        print(f"Need to download {needed} more images.")
        
        downloaded_valid = 0
        current_idx = start_count + 1
        
        keyword_idx = 0
        while downloaded_valid < needed:
            if keyword_idx >= len(keywords):
                print(f"Warning: Cycle through all keywords completed but only got {downloaded_valid}/{needed} images.")
                keyword = f"{class_name} waste garbage"
            else:
                keyword = keywords[keyword_idx]
                keyword_idx += 1
                
            print(f"\nSearching Bing for keyword: '{keyword}'...")
            
            # Clean/create temp dir
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Request some buffer images
            limit_to_download = max(15, (needed - downloaded_valid) + 5)
            
            try:
                # Use Downloader directly to avoid on_image hook progress-bar error
                dl.search(
                    query=keyword,
                    limit=limit_to_download,
                    output_dir=temp_dir,
                    engine="bing",
                    adult_filter_off=True,
                    force_replace=True,
                    timeout=10,
                    verbose=False
                )
            except Exception as e:
                print(f"Downloader failed for '{keyword}': {e}")
                time.sleep(5)
                continue
                
            # Find the downloaded files
            subdirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            if not subdirs:
                print("No directories created in temp downloads.")
                continue
                
            search_subdir = os.path.join(temp_dir, subdirs[0])
            downloaded_files = sorted([f for f in os.listdir(search_subdir) if os.path.isfile(os.path.join(search_subdir, f))])
            print(f"Downloaded {len(downloaded_files)} candidate images. Validating...")
            
            for file_name in downloaded_files:
                if file_name.endswith('.json') or file_name.endswith('.jsonl') or file_name.startswith('_'):
                    continue
                src_path = os.path.join(search_subdir, file_name)
                
                dest_filename = f"{current_idx:02d}.jpg"
                dest_path = os.path.join(class_dir, dest_filename)
                
                try:
                    with Image.open(src_path) as img:
                        img.verify()
                    with Image.open(src_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(dest_path, 'JPEG')
                    
                    print(f"Saved: {dest_filename}")
                    downloaded_valid += 1
                    current_idx += 1
                    
                    if downloaded_valid >= needed:
                        break
                except Exception as img_err:
                    pass
            
            print(f"Status for {class_name}: {downloaded_valid}/{needed} new images downloaded.")
            time.sleep(3)
            
        print(f"Finished class {class_name}. Class folder now has {get_existing_count(class_dir)} images.")
        time.sleep(3)
        
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print("\nAll classes successfully processed!")

if __name__ == "__main__":
    main()

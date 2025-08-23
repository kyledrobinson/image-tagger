import os
import csv
import re
import time
from PIL import Image
from PIL.ExifTags import TAGS
from collections import Counter
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import torch
import webcolors
from labels import label_list  # Import expanded label list

# 🚨 CUDA Check
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("🚀 CUDA is available. Using GPU!")
    print(f"🚀 Device Name: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    reserved_mem = torch.cuda.memory_reserved(0) / (1024 ** 3)
    allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"🧠 Total Memory: {total_mem:.2f} GB")
    print(f"🧠 Reserved Memory: {reserved_mem:.2f} GB")
    print(f"🧠 Allocated Memory: {allocated_mem:.2f} GB")
else:
    print("⚠️ CUDA not found. Using CPU.")

# Initialize CLIP and BLIP
print("Loading CLIP and BLIP models...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

print("Generating CLIP label embeddings...")
label_inputs = clip_processor(text=label_list, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    label_features = clip_model.get_text_features(**label_inputs)
    label_features = label_features / label_features.norm(p=2, dim=-1, keepdim=True)

# Helper functions
def extract_keywords(caption):
    words = re.findall(r'\b\w+\b', caption.lower())
    return list(set(w for w in words if len(w) > 3))

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_dominant_color(image_path):
    try:
        img = Image.open(image_path).convert('RGB').resize((50, 50))
        result = img.getcolors(50 * 50)
        dominant_color = sorted(result, key=lambda x: x[0], reverse=True)[0][1]
        hex_color = webcolors.rgb_to_hex(dominant_color)
        color_name = closest_color(dominant_color)
    except Exception as e:
        print(f"Color extraction error for {image_path}: {e}")
        hex_color, color_name = "Error", "Unknown"
    return hex_color, color_name

# Get image folder
image_folder = input("Enter the path to your image folder: ")
output_csv = "image_tags_with_clip_blip.csv"

# Load processed files
processed_files_set = set()
if os.path.exists(output_csv):
    with open(output_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        processed_files_set = {row[0] for row in reader}

# Find new files to process
all_files = []
for root, dirs, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')) and file not in processed_files_set:
            all_files.append((root, file))

total_files = len(all_files)
if total_files == 0:
    print("✅ No new files to process. Exiting.")
    exit()

print(f"🔎 Found {total_files} new image(s) to process.")

start_time = time.time()
processed_count = 0

with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    if not processed_files_set:
        writer.writerow(['Filename', 'Filename_Tags', 'Metadata_Tags', 'CLIP_Best_Labels', 'BLIP_Caption', 'BLIP_Keywords', 'Hex_Color', 'Color_Name'])

    pbar = tqdm(all_files, desc="Processing images", unit="img",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] ({postfix})")

    for root, file in pbar:
        filepath = os.path.join(root, file)
        filename_tags = os.path.splitext(file)[0].replace('_', ' ').replace('-', ' ')
        metadata_tags = []
        
        try:
            image = Image.open(filepath)
            exif_data = image._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in ['XPKeywords', 'ImageDescription', 'UserComment']:
                        metadata_tags.append(str(value))
        except Exception as e:
            print(f"Metadata error: {e}")

        try:
            image_clip = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_clip)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                similarity = torch.matmul(image_features, label_features.T)
                best_match_idx = similarity[0].topk(3).indices
                best_labels = [label_list[i] for i in best_match_idx]
        except Exception as e:
            print(f"CLIP error: {e}")
            best_labels = ["CLIP Error"]

        try:
            raw_image = blip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = blip_model.generate(**raw_image)
                caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
                blip_keywords = extract_keywords(caption)
        except Exception as e:
            print(f"BLIP error: {e}")
            caption = "BLIP Error"
            blip_keywords = []

        hex_color, color_name = get_dominant_color(filepath)

        writer.writerow([file, filename_tags, ', '.join(metadata_tags), ', '.join(best_labels), caption, ', '.join(blip_keywords), hex_color, color_name])
        processed_count += 1
        images_left = total_files - processed_count
        pbar.set_postfix_str(f"{images_left} images left")

elapsed = time.time() - start_time
print(f"\n🎉 Tagging complete! Processed {processed_count} new images in {elapsed:.2f} seconds.")

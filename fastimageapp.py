from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import colorsys
import json
import os
import shutil
from typing import List, Dict, Any, Optional
import uuid
import time
import re

# Note: ImageTagger and ImageMetadata are imported inside get_image_tagger() for lazy loading

app = FastAPI()

# Create necessary directories
os.makedirs("static/images", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize the image tagger (lazy loading)
image_tagger = None

def get_image_tagger():
    global image_tagger
    if image_tagger is None:
        try:
            print("🔄 Loading AI models (CLIP + BLIP)... This may take 30-60 seconds...")
            # Import here to avoid startup crashes
            from clip_tagger import ImageTagger, ImageMetadata
            image_tagger = ImageTagger()
            print("✅ AI models loaded successfully!")
        except Exception as e:
            print(f"Error loading AI models: {e}")
            raise HTTPException(status_code=500, detail="Error loading AI models")
    return image_tagger

# Serve static files (images and XMP)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def test_endpoint():
    return {"status": "Server is working!", "message": "FastAPI is running correctly"}

# Root route to serve the HTML template
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("complete-test.html", {"request": request})

@app.get("/api/images")
async def list_images(
    directory: str = "static/images",
    search: Optional[str] = None,
    tag: Optional[str] = None,
    format: Optional[str] = None,
    sort_by: str = "uploaded_at",
    sort_order: str = "desc"
):
    """List all images with their metadata, with optional search and sort functionality"""
    image_dir = Path(directory)
    images = []
    
    # Create directory if it doesn't exist
    image_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_dir.glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                # Get basic file info only (no AI processing for existing images)
                stat = img_path.stat()
                
                # Try to read basic image dimensions without AI models
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        width, height = img.size
                        format_name = img.format
                except:
                    width, height = None, None
                    format_name = img_path.suffix[1:].upper()
                
                # Basic metadata without AI processing
                metadata_dict = {
                    "caption": img_path.stem.replace('_', ' ').title(),
                    "tags": ["existing"],  # Mark as existing image
                    "all_tags": [],
                    "colors": [],
                    "dimensions": f"{width}x{height}" if width and height else "",
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "format": format_name,
                }

                # Try to load existing colors from XMP sidecar (no AI processing)
                try:
                    xmp_path = img_path.with_suffix('.xmp')
                    if xmp_path.exists():
                        with open(xmp_path, 'r', encoding='utf-8') as xf:
                            xmp_text = xf.read()
                        if 'xmpDM:ColorantSwatchList' in xmp_text:
                            section = xmp_text.split('xmpDM:ColorantSwatchList', 1)[1].split('</xmpDM:ColorantSwatchList>', 1)[0]
                            # Find each swatch entry
                            swatches = re.findall(r'<rdf:li[^>]*>.*?</rdf:li>', section, re.DOTALL)
                            colors = []
                            for sw in swatches:
                                hex_match = re.search(r'xmpDM:value[\s=\"]+[#]?([0-9A-Fa-f]{6})', sw)
                                perc_match = re.search(r'\((\d+\.?\d*)%\)', sw)
                                if not hex_match:
                                    continue
                                hex_code = '#' + hex_match.group(1).upper()
                                # Convert to RGB and HLS for shade/temperature inference
                                r = int(hex_code[1:3], 16)
                                g = int(hex_code[3:5], 16)
                                b = int(hex_code[5:7], 16)
                                h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
                                h_deg = round(h * 360, 1)
                                s_pct = round(s * 100, 1)
                                l_pct = round(l * 100, 1)
                                # Infer temperature (simple heuristic)
                                if r > b + 30 and g > b + 30:
                                    temperature = 'warm'
                                elif b > r + 30 and (b > g + 10):
                                    temperature = 'cool'
                                else:
                                    temperature = 'neutral'
                                # Infer shade focusing on gray detect so search works
                                if l_pct < 15:
                                    shade = 'black'
                                elif l_pct > 90:
                                    shade = 'white'
                                elif s_pct < 15:
                                    shade = 'gray'
                                else:
                                    shade = 'unknown'
                                # Percentage if present
                                percentage = float(perc_match.group(1)) if perc_match else 0.0
                                colors.append({
                                    'hex': hex_code,
                                    'rgb': {'r': r, 'g': g, 'b': b},
                                    'hsl': {'h': h_deg, 's': s_pct, 'l': l_pct},
                                    'name': shade if shade in ('black','white','gray') else 'unknown',
                                    'percentage': percentage,
                                    'temperature': temperature,
                                    'shade': shade
                                })
                            if colors:
                                metadata_dict["colors"] = colors
                except Exception as _e:
                    # Non-fatal; keep colors empty if parsing fails
                    pass
                
                # Prepare the response
                image_data = {
                    "filename": img_path.name,
                    "path": f"/static/images/{img_path.name}",
                    "uploaded_at": stat.st_mtime,
                    "metadata": metadata_dict
                }
                images.append(image_data)
                
            except Exception as e:
                print(f"Error reading basic info for {img_path}: {e}")
                # Fallback to minimal info
                images.append({
                    "filename": img_path.name,
                    "path": f"/static/images/{img_path.name}",
                    "uploaded_at": img_path.stat().st_mtime,
                    "metadata": {
                        "caption": img_path.stem.replace('_', ' ').title(),
                        "tags": ["error"],
                        "all_tags": [],
                        "colors": [],
                        "error": str(e)
                    }
                })
    
    # Apply search filters
    if search:
        search_lower = search.lower()
        filtered_images = []
        for img in images:
            # Search in filename, caption, and tags
            # Also include color names, shades, hex values, and temperature
            colors = img["metadata"].get("colors", []) or []
            color_terms = []
            for c in colors:
                name = str(c.get("name", "")).lower()
                shade = str(c.get("shade", "")).lower()
                hexv = str(c.get("hex", "")).lower()
                temp = str(c.get("temperature", "")).lower()
                perc = c.get("percentage")
                # Add terms (gate name/shade on percentage to reduce false positives)
                perc_ok = False
                try:
                    if perc is not None and float(perc) >= 1.0:
                        perc_ok = True
                except Exception:
                    pass
                if name and perc_ok:
                    color_terms.append(name)
                if shade and perc_ok:
                    color_terms.append(shade)
                if hexv:
                    color_terms.append(hexv)
                if temp:
                    color_terms.append(temp)
                # Add percentage terms (support queries like "5" and "5%")
                try:
                    if perc is not None:
                        # normalize to one decimal to match typical display like 5.0
                        perc_float = float(perc)
                        perc_int = int(round(perc_float))
                        color_terms.append(str(perc_int))
                        color_terms.append(f"{perc_int}%")
                        # also include one-decimal variants if different from int
                        perc_1 = round(perc_float, 1)
                        if perc_1 != perc_int:
                            color_terms.append(str(perc_1))
                            color_terms.append(f"{perc_1}%")
                except Exception:
                    pass
                # Add grey/gray synonym forms to improve matching
                for term in [name, shade]:
                    if term and perc_ok:
                        color_terms.append(term.replace("gray", "grey"))
                        color_terms.append(term.replace("grey", "gray"))

            searchable_text = (
                img["filename"].lower() + " " +
                img["metadata"].get("caption", "").lower() + " " +
                " ".join(img["metadata"].get("tags", [])).lower() + " " +
                " ".join(img["metadata"].get("all_tags", [])).lower() + " " +
                " ".join(color_terms)
            )
            if search_lower in searchable_text:
                filtered_images.append(img)
        images = filtered_images
    
    # Apply tag filter
    if tag:
        tag_lower = tag.lower()
        images = [
            img for img in images 
            if any(tag_lower in t.lower() for t in img["metadata"].get("tags", []) + img["metadata"].get("all_tags", []))
        ]
    
    # Apply format filter
    if format:
        format_lower = format.lower()
        images = [
            img for img in images 
            if img["metadata"].get("format", "").lower() == format_lower
        ]
    
    # Apply sorting
    valid_sort_fields = ["filename", "uploaded_at", "size_mb", "format"]
    if sort_by not in valid_sort_fields:
        sort_by = "uploaded_at"
    
    reverse_order = sort_order.lower() == "desc"
    
    if sort_by == "filename":
        images.sort(key=lambda x: x["filename"].lower(), reverse=reverse_order)
    elif sort_by == "uploaded_at":
        images.sort(key=lambda x: x.get("uploaded_at", 0), reverse=reverse_order)
    elif sort_by == "size_mb":
        images.sort(key=lambda x: x["metadata"].get("size_mb", 0), reverse=reverse_order)
    elif sort_by == "format":
        images.sort(key=lambda x: x["metadata"].get("format", "").lower(), reverse=reverse_order)
    
    return {
        "images": images,
        "total": len(images),
        "filters_applied": {
            "search": search,
            "tag": tag,
            "format": format,
            "sort_by": sort_by,
            "sort_order": sort_order
        }
    }

@app.post("/api/images")
async def upload_image(file: UploadFile = File(...), preserve_filename: bool = True, custom_filename: str = None):
    """Upload and process a new image with filename control"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Handle filename based on user preference
        file_extension = Path(file.filename).suffix
        
        if custom_filename:
            # Use custom filename (sanitize it)
            safe_custom = re.sub(r'[^\w\-_\.]', '_', custom_filename)
            if not safe_custom.endswith(file_extension):
                safe_custom += file_extension
            chosen_filename = safe_custom
        elif preserve_filename:
            # Keep original filename, handle duplicates
            original_name = Path(file.filename).stem
            safe_original = re.sub(r'[^\w\-_\.]', '_', original_name)
            chosen_filename = f"{safe_original}{file_extension}"
            
            # Handle duplicates by adding number
            counter = 1
            base_path = Path("static/images")
            while (base_path / chosen_filename).exists():
                chosen_filename = f"{safe_original}_{counter}{file_extension}"
                counter += 1
        else:
            # Use UUID (original behavior)
            chosen_filename = f"{uuid.uuid4()}{file_extension}"
        
        file_path = Path("static/images") / chosen_filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📁 Saved image: {file_path} (original: {file.filename})")
        
        # Process the image with CLIP and BLIP
        print(f"🤖 Processing image with AI models...")
        tagger = get_image_tagger()
        if tagger is None:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "filename": chosen_filename,
                    "original_filename": file.filename,
                    "path": f"/static/images/{chosen_filename}",
                    "error": "AI models not available - check server logs",
                    "metadata": {
                        "caption": Path(chosen_filename).stem.replace('_', ' ').title(),
                        "tags": ["error", "no_ai"]
                    }
                }
            )
        
        result = tagger.process_image(str(file_path))
        print(f"🔍 DEBUG: AI processing result: {result}")
        
        if "error" in result:
            # Don't delete the file, just return the error
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "filename": chosen_filename,
                    "original_filename": file.filename,
                    "path": f"/static/images/{chosen_filename}",
                    "error": result["error"],
                    "metadata": {
                        "caption": Path(chosen_filename).stem.replace('_', ' ').title(),
                        "tags": ["error"]
                    }
                }
            )
        
        # Return the AI-generated metadata directly (clip_tagger returns data in correct format)
        return {
            "status": "success",
            "filename": chosen_filename,
            "original_filename": file.filename,
            "path": f"/static/images/{chosen_filename}",
            "metadata": {
                "caption": result.get("caption", Path(chosen_filename).stem.replace('_', ' ').title()),
                "tags": result.get("tags", ["untagged"]),
                "all_tags": result.get("all_tags", []),
                "colors": result.get("colors", []),
                "dimensions": result.get("dimensions", ""),
                "size_mb": result.get("size_mb", 0),
                "format": result.get("format", "")
            }
        }
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        # Clean up the file if there was an error
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/api/upload")
async def upload_image_alt(file: UploadFile = File(...), preserve_filename: bool = True, custom_filename: str = None):
    """Alternative upload endpoint for frontend compatibility"""
    return await upload_image(file, preserve_filename, custom_filename)

@app.get("/api/images/{filename}/reprocess")
async def reprocess_image(filename: str):
    """Reprocess an existing image to update its metadata"""
    try:
        file_path = Path("static/images") / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        print(f"Reprocessing image: {file_path}")
        
        # Process the image with CLIP and BLIP
        result = get_image_tagger().process_image(str(file_path))
        
        if "error" in result:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "filename": filename,
                    "path": f"/static/images/{filename}",
                    "error": result["error"]
                }
            )
        
        return {
            "status": "success",
            "filename": filename,
            "path": f"/static/images/{filename}",
            "metadata": result.get("metadata", {
                "caption": Path(filename).stem.replace('_', ' ').title(),
                "tags": ["untagged"]
            })
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error reprocessing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reprocessing image: {str(e)}")

@app.delete("/api/images/{filename}")
async def delete_image(filename: str):
    """Delete an image and its metadata"""
    try:
        image_path = Path("static/images") / filename
        xmp_path = image_path.with_suffix(".xmp")
        
        # Delete the image and its XMP sidecar if they exist
        if image_path.exists():
            image_path.unlink()
        if xmp_path.exists():
            xmp_path.unlink()
            
        return {"status": "success", "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    print("📁 Current working directory:", os.getcwd())
    print("📂 Templates directory exists:", os.path.exists("templates"))
    print("📂 Static directory exists:", os.path.exists("static"))
    print("📄 complete-test.html exists:", os.path.exists("templates/complete-test.html"))
    
    # Test clip_tagger import before starting server
    try:
        print("🧪 Testing clip_tagger import...")
        from clip_tagger import ImageTagger, ImageMetadata
        print("✅ clip_tagger import successful")
    except Exception as e:
        print(f"❌ clip_tagger import failed: {e}")
        print("⚠️  Server will start but AI processing won't work")
    
    try:
        print("🌐 Starting Uvicorn server...")
        uvicorn.run("fastimageapp:app", host="127.0.0.1", port=8000, reload=False)
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
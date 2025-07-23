import os
import json
import torch
import numpy as np
import webcolors
import piexif
from PIL import Image, ImageFile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import colorsys
from datetime import datetime
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    BlipProcessor, 
    BlipForConditionalGeneration
)
import re
import cv2
from sklearn.cluster import KMeans

# Image processing settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

@dataclass
class ImageMetadata:
    path: str
    caption: str = ""
    tags: List[str] = None
    clip_scores: Dict[str, float] = None
    width: int = 0
    height: int = 0
    format: str = ""
    size_mb: float = 0.0
    dominant_colors: List[Dict[str, Any]] = None

class ImageTagger:
    def __init__(self, device: str = None):
        """Initialize the image tagger with CLIP and BLIP models."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        
        # Define basic colors and precompute RGB values for faster color matching
        self.BASIC_COLORS = {
            '#FF0000': 'red', '#00FF00': 'lime', '#0000FF': 'blue',
            '#FFFF00': 'yellow', '#00FFFF': 'cyan', '#FF00FF': 'magenta',
            '#800000': 'maroon', '#808000': 'olive', '#008000': 'green',
            '#800080': 'purple', '#008080': 'teal', '#000080': 'navy',
            '#C0C0C0': 'silver', '#808080': 'gray', '#000000': 'black',
            '#FFFFFF': 'white', '#FFA500': 'orange', '#A52A2A': 'brown',
            '#FFC0CB': 'pink', '#EE82EE': 'violet', '#40E0D0': 'turquoise',
            '#F5F5DC': 'beige', '#F5DEB3': 'wheat', '#2E8B57': 'sea green',
            '#D2691E': 'chocolate', '#6495ED': 'cornflower blue', '#DC143C': 'crimson',
            '#B22222': 'firebrick', '#FFD700': 'gold', '#DAA520': 'goldenrod',
            '#90EE90': 'light green', '#FFA07A': 'light salmon', '#20B2AA': 'light sea green',
            '#87CEFA': 'light sky blue', '#778899': 'light slate gray', '#B0C4DE': 'light steel blue',
            '#FFB6C1': 'light pink', '#32CD32': 'lime green',
            '#66CDAA': 'medium aquamarine', '#0000CD': 'medium blue', '#BA55D3': 'medium orchid',
            '#9370DB': 'medium purple', '#3CB371': 'medium sea green', '#7B68EE': 'medium slate blue',
            '#00FA9A': 'medium spring green', '#48D1CC': 'medium turquoise',
            '#C71585': 'medium violet red', '#191970': 'midnight blue', '#FF4500': 'orange red',
            '#DA70D6': 'orchid', '#EEE8AA': 'pale goldenrod',
            '#98FB98': 'pale green', '#AFEEEE': 'pale turquoise', '#DB7093': 'pale violet red',
            '#FFEFD5': 'papaya whip', '#FFDAB9': 'peach puff', '#CD853F': 'peru',
            '#DDA0DD': 'plum', '#B0E0E6': 'powder blue',
            '#663399': 'rebecca purple',
            '#BC8F8F': 'rosy brown', '#4169E1': 'royal blue', '#8B4513': 'saddle brown',
            '#FA8072': 'salmon', '#F4A460': 'sandy brown',
            '#FFF5EE': 'seashell', '#A0522D': 'sienna',
            '#87CEEB': 'sky blue', '#6A5ACD': 'slate blue', '#708090': 'slate gray',
            '#FFFAFA': 'snow', '#00FF7F': 'spring green', '#4682B4': 'steel blue',
            '#D2B48C': 'tan', '#D8BFD8': 'thistle',
            '#FF6347': 'tomato', '#40E0D0': 'turquoise',
            '#F5DEB3': 'wheat', '#F5F5F5': 'white smoke',
            '#9ACD32': 'yellow green'
        }
        
        # Precompute RGB values for faster color matching
        self._color_cache = {}
        for hex_code, name in self.BASIC_COLORS.items():
            self._color_cache[hex_code] = {
                'name': name,
                'rgb': self._hex_to_rgb(hex_code)
            }
        
        # Clean and deduplicate candidate labels
        self.candidate_labels = list(dict.fromkeys([
            # General categories
            "portrait", "landscape", "nature", "city", "animal", "food", "indoor", "outdoor",
            "person", "people", "building", "car", "tree", "water", "mountain", "sky", "sunset",
            "beach", "snow", "rain", "night", "day", "sunny", "cloudy", "winter", "summer",
            "spring", "autumn", "art", "abstract", "modern", "vintage", "black and white", "colorful",
            
            # Specific animals
            "cat", "kitten", "feline", "dog", "puppy", "canine", "bird", "eagle", "sparrow", 
            "fish", "goldfish", "shark", "reptile", "snake", "lizard", "turtle", "amphibian", 
            "frog", "toad", "insect", "butterfly", "bee", "ant", "spider", "wildlife", "mammal",
            
            # Plants and nature
            "flower", "rose", "tulip", "daisy", "oak", "pine", "maple", "grass", "forest", 
            "jungle", "garden", "park", "meadow", "field", "hill", "valley",
            
            # Water bodies
            "ocean", "sea", "lake", "river", "stream", "waterfall", "pond", "puddle", "shore",
            
            # Urban and man-made
            "skyscraper", "house", "home", "apartment", "office", "school", "hospital",
            "restaurant", "cafe", "shop", "store", "market", "mall", "stadium", "airport", "station",
            "road", "street", "highway", "bridge", "tunnel", "monument", "statue", "sculpture",
            
            # Transportation
            "automobile", "truck", "bus", "motorcycle", "bicycle", "bike", "scooter",
            "airplane", "helicopter", "boat", "ship", "yacht", "train", "subway", "tram",
            
            # People
            "man", "woman", "boy", "girl", "baby", "toddler", "child", "teenager", "adult", "elderly",
            "family", "friends", "couple", "group", "crowd", "human",
            
            # Activities
            "sports", "soccer", "football", "basketball", "tennis", "swimming", "running", "walking",
            "dancing", "singing", "playing", "working", "cooking", "eating", "drinking", "sleeping",
            "reading", "writing", "painting", "drawing", "photography", "filming", "traveling", "hiking",
            
            # Objects
            "furniture", "chair", "table", "desk", "bed", "sofa", "couch", "lamp", "light", "computer",
            "laptop", "phone", "smartphone", "camera", "book", "notebook", "pen", "pencil", "paper",
            "clothing", "shirt", "pants", "dress", "hat", "shoes", "sneakers", "boots", "jacket",
            "bag", "backpack", "purse", "wallet", "watch", "jewelry", "ring", "necklace", "bracelet"
        ]))

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _find_closest_color(self, hex_color):
        """Find the closest color name from our BASIC_COLORS dictionary"""
        try:
            hex_upper = hex_color.upper()
            if hex_upper in self.BASIC_COLORS:
                return self.BASIC_COLORS[hex_upper]
            
            # Otherwise find the closest color by RGB distance
            r1, g1, b1 = self._hex_to_rgb(hex_upper)
            min_distance = float('inf')
            closest_color = hex_upper  # Default to hex if no close match found
            
            for hex_code, color_data in self._color_cache.items():
                r2, g2, b2 = color_data['rgb']
                # Fast approximate distance calculation (no sqrt needed for comparison)
                distance = (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_data['name']
                    
            return closest_color
        except Exception:
            return hex_upper  # Return original hex if any error occurs

    def _get_color_shade(self, rgb: Tuple[int, int, int], hue: float, lightness: float, saturation: float) -> str:
        """
        Determine the shade of a color based on its properties.
        
        Args:
            rgb: Tuple of (r, g, b) values (0-255)
            hue: Hue angle in degrees (0-360)
            lightness: Lightness percentage (0-100)
            saturation: Saturation percentage (0-100)
            
        Returns:
            str: The determined shade name
        """
        try:
            r, g, b = rgb
            
            # Very dark colors
            if lightness < 15:
                return 'black'
                
            # Very light colors
            if lightness > 90:
                return 'white'
                
            # Grays (low saturation)
            if saturation < 15:
                if lightness > 80:
                    return 'white'
                elif lightness > 60:
                    return 'light gray'
                elif lightness > 40:
                    return 'gray'
                else:
                    return 'dark gray'
            
            # Get the color name from our basic colors
            hex_color = f"#{r:02x}{g:02x}{b:02x}".upper()
            color_name = self._find_closest_color(hex_color)
            
            # Add shade modifier based on lightness
            if lightness > 80:
                return f"light {color_name}" if color_name != 'white' else 'white'
            elif lightness > 60:
                return color_name if color_name != 'gray' else 'medium gray'
            elif lightness > 40:
                return f"dark {color_name}" if color_name != 'black' else 'black'
            else:
                return f"dark {color_name}" if color_name != 'black' else 'black'
                
        except Exception as e:
            print(f"Error determining color shade: {e}")
            return 'unknown'

    def load_models(self):
        """Load the CLIP and BLIP models with optimized settings."""
        if not self.clip_model:
            # Load CLIP model
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            
        if not self.blip_model:
            # Load BLIP model with fast processor
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.blip_model = self.blip_model.to(self.device)
            
        # Set models to evaluation mode
        self.clip_model.eval()
        self.blip_model.eval()
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess an image with optimizations"""
        try:
            # Load image in the most efficient way
            with Image.open(image_path) as img:
                # Convert to RGB if needed (faster than checking mode first)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize large images for faster processing (more aggressive)
                max_size = 512  # Reduced from 1024 for faster processing
                if max(img.size) > max_size:
                    # Use faster resampling method
                    img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
                    print(f"🔧 Resized image to {img.size} for faster processing")
                
                return img.copy()  # Make a copy since we're using 'with' statement
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def generate_caption(self, image: Image.Image) -> str:
        """Generate a caption for the image using BLIP."""
        if not self.blip_model or not self.blip_processor:
            self.load_models()
            
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
            
        caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
    
    def get_dominant_colors(self, image: Image.Image, n_colors: int = 5) -> List[Dict[str, Any]]:
        """
        Extract dominant colors from an image using k-means clustering.
        Returns a list of color dictionaries with hex, RGB, HSL, name info, and pixel percentages.
        """
        def rgb_to_hex(r, g, b):
            """Convert RGB values to hex color code"""
            return f"#{r:02x}{g:02x}{b:02x}"

        try:
            # Convert image to numpy array and reshape for k-means
            img_array = np.array(image)
            
            # Handle RGBA images by converting to RGB
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
            
            # Reshape to a 2D array of pixels (height*width, 3)
            h, w, c = img_array.shape
            img_array = img_array.reshape((h * w, c))
            
            # Remove any transparent/alpha pixels if they exist
            if img_array.shape[1] == 4:
                img_array = img_array[img_array[..., 3] > 0, :3]
            
            # Skip if no valid pixels found
            if len(img_array) == 0:
                return [{
                    'hex': '#000000',
                    'rgb': (0, 0, 0),
                    'hsl': (0, 0, 0),
                    'name': 'black',
                    'percentage': 100.0,
                    'brightness': 0,
                    'temperature': 'neutral',
                    'shade': 'black'
                }]
            
            # Store total pixel count for percentage calculation
            total_pixels = len(img_array)
            
            # Use k-means to find dominant colors
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(img_array)
            
            # Get the cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Get the number of pixels in each cluster
            counts = np.bincount(kmeans.labels_)
            
            # Sort colors by frequency (most common first)
            sorted_indices = np.argsort(counts)[::-1]
            colors = colors[sorted_indices]
            counts = counts[sorted_indices]  # Also sort the counts to match
            
            result = []
            for i, color in enumerate(colors):
                # Ensure color values are in valid range
                r, g, b = np.clip(color, 0, 255).astype(int)
                
                # Calculate percentage of pixels this color represents
                pixel_percentage = (counts[i] / total_pixels) * 100
                
                # Convert to hex
                hex_color = rgb_to_hex(r, g, b)
                
                # Convert to HSL
                h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
                h = round(h * 360, 1)
                s = round(s * 100, 1)
                l = round(l * 100, 1)
                
                # Get color name
                color_name = self._find_closest_color(hex_color)
                
                # Calculate brightness (0-1)
                brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                
                # Determine color temperature
                if r > b + 30 and g > b + 30:  # More red and green than blue
                    temperature = 'warm'
                elif b > r + 30 and (b > g + 10):  # More blue than red and green
                    temperature = 'cool'
                else:
                    temperature = 'neutral'
                
                # Determine color shade
                if brightness < 0.2:
                    shade = 'black'
                elif brightness > 0.8:
                    shade = 'white'
                elif s < 15:  # Low saturation
                    shade = 'gray'
                else:
                    shade = color_name
                
                result.append({
                    'hex': hex_color,
                    'rgb': (r, g, b),
                    'hsl': (h, s, l),
                    'name': color_name,
                    'percentage': round(pixel_percentage, 1),
                    'brightness': round(brightness, 2),
                    'temperature': temperature,
                    'shade': shade
                })
            
            return result
            
        except Exception as e:
            print(f"Error in color analysis: {str(e)}")
            # Return a default color if analysis fails
            return [{
                'hex': '#000000',
                'rgb': (0, 0, 0),
                'hsl': (0, 0, 0),
                'name': 'black',
                'percentage': 100.0,
                'brightness': 0,
                'temperature': 'neutral',
                'shade': 'black'
            }]
    
    def get_tags(self, image: Image.Image, candidate_labels: List[str] = None) -> List[Dict[str, Any]]:
        """Get relevant tags for the image using CLIP."""
        if not self.clip_model or not self.clip_processor:
            self.load_models()
        
        # Use provided labels or default ones
        labels = candidate_labels or self.candidate_labels
        
        # Process image and text
        inputs = self.clip_processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            
        # Get similarity scores
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Print top 10 labels for debugging
        print("\nTop 10 labels for image:")
        top_indices = np.argsort(probs)[-10:][::-1]
        for idx in top_indices:
            print(f"  {labels[idx]}: {probs[idx]:.2%}")
        
        # Combine labels with scores and filter by threshold
        tags = [
            {
                "tag": label, 
                "score": float(score), 
                "percentage": float(score * 100)
            }
            for label, score in zip(labels, probs)
            if score > 0.05  # Lowered threshold from 0.1 to 0.05 (5%)
        ]
        
        # Sort by score in descending order
        tags.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top 20 tags to keep the output manageable
        return tags[:20]
    
    def write_metadata_to_image(self, image_path: str, metadata: ImageMetadata) -> Tuple[bool, List[str]]:
        """Write metadata to image file and create XMP sidecar."""
        modified_files = []
        try:
            img_path = Path(image_path)
            img = Image.open(img_path)
            
            # Create XMP sidecar
            xmp_path = img_path.with_suffix('.xmp')
            xmp_content = [
                '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>',
                '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core">',
                '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
                '           xmlns:dc="http://purl.org/dc/elements/1.1/"',
                '           xmlns:xmp="http://ns.adobe.com/xap/1.0/"',
                '           xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/">',
                '    <rdf:Description rdf:about="">'
            ]
            
            # Add basic metadata
            if metadata.caption:
                xmp_content.extend([
                    f'      <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{metadata.caption}</rdf:li></rdf:Alt></dc:description>',
                    f'      <xmp:Description>{metadata.caption}</xmp:Description>'
                ])
                
            # Add tags/keywords
            if metadata.tags:
                xmp_content.append('      <dc:subject><rdf:Bag>')
                xmp_content.extend(f'        <rdf:li>{tag}</rdf:li>' for tag in metadata.tags)
                xmp_content.append('      </rdf:Bag></dc:subject>')
                xmp_content.append(f'      <photoshop:Keywords><rdf:Bag><rdf:li>{", ".join(metadata.tags)}</rdf:li></rdf:Bag></photoshop:Keywords>')
            
            # Add dominant colors if available
            if metadata.dominant_colors:
                xmp_content.append('      <xmpDM:ColorantSwatchList xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/">')
                for i, color in enumerate(metadata.dominant_colors[:5], 1):
                    hex_color = color['hex'].lstrip('#')
                    xmp_content.extend([
                        f'        <rdf:Bag>',
                        f'          <rdf:li rdf:parseType="Resource">',
                        f'            <xmpDM:swatchName>Color {i} ({color["percentage"]}%)</xmpDM:swatchName>',
                        f'            <xmpDM:value>#{hex_color}</xmpDM:value>',
                        f'          </rdf:li>',
                        f'        </rdf:Bag>'
                    ])
                xmp_content.append('      </xmpDM:ColorantSwatchList>')
            
            # Close XMP structure
            xmp_content.extend([
                '    </rdf:Description>',
                '  </rdf:RDF>',
                '</x:xmpmeta>',
                '<?xpacket end="w"?>'
            ])
            
            # Write XMP file
            with open(xmp_path, "w", encoding="utf-8") as f:
                f.write('\n'.join(xmp_content))
            modified_files.append(str(xmp_path))
            
            # Update EXIF data for JPEG/TIFF
            if img.format in {"JPEG", "TIFF"}:
                try:
                    exif_dict = piexif.load(img.info.get("exif", b""))
                    
                    # Update standard EXIF fields
                    if metadata.caption:
                        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = metadata.caption.encode("utf-8")
                        exif_dict["0th"][piexif.ImageIFD.XPTitle] = metadata.caption.encode("utf-16")
                        
                    if metadata.tags:
                        keywords = ", ".join(metadata.tags).encode("utf-8")
                        exif_dict["0th"][piexif.ImageIFD.XPKeywords] = keywords
                        
                    # Save updated EXIF
                    exif_bytes = piexif.dump(exif_dict)
                    img.save(img_path, exif=exif_bytes)
                    modified_files.append(str(img_path))
                    
                except Exception as e:
                    print(f"Warning: Could not update EXIF data: {e}")
            
            return True, modified_files
            
        except Exception as e:
            print(f"Error writing metadata: {e}")
            return False, modified_files
        finally:
            if 'img' in locals():
                img.close()
    
    def read_metadata(self, image_path: str) -> ImageMetadata:
        """Read metadata from image file and XMP sidecar."""
        try:
            img_path = Path(image_path)
            img = Image.open(img_path)
            
            # Get basic image info
            width, height = img.size
            img_format = img.format or ""
            size_mb = os.path.getsize(image_path) / (1024 * 1024)
            
            # Initialize metadata with basic info
            metadata = ImageMetadata(
                path=str(img_path),
                width=width,
                height=height,
                format=img_format,
                size_mb=round(size_mb, 2),
                tags=[],
                dominant_colors=[]
            )
            
            # Try to read XMP sidecar first
            xmp_path = img_path.with_suffix('.xmp')
            if xmp_path.exists():
                try:
                    with open(xmp_path, 'r', encoding='utf-8') as f:
                        xmp_content = f.read()
                    
                    # Parse XMP content for caption
                    if '<dc:description>' in xmp_content:
                        start = xmp_content.find('<dc:description>') + len('<dc:description>')
                        end = xmp_content.find('</dc:description>', start)
                        if start > 0 and end > start:
                            metadata.caption = xmp_content[start:end].strip()
                    
                    # Parse tags - improved to handle RDF containers properly
                    if '<dc:subject>' in xmp_content:
                        # Extract the entire subject section
                        subject_section = xmp_content.split('<dc:subject>', 1)[1].split('</dc:subject>', 1)[0]
                        
                        # Clean up the tags - remove RDF container tags and other XML/HTML tags
                        import re
                        
                        # Remove all XML/HTML tags
                        clean_text = re.sub(r'<[^>]+>', ' ', subject_section)
                        
                        # Clean up any remaining special characters and normalize whitespace
                        clean_text = re.sub(r'[\n\r\t]', ' ', clean_text)  # Remove newlines and tabs
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
                        
                        # Split into tags and clean each one
                        raw_tags = [t.strip() for t in clean_text.split() if t.strip()]
                        
                        # Filter out RDF and other system tags
                        metadata.tags = [
                            tag for tag in raw_tags 
                            if tag.lower() not in {'rdf:li', 'rdf:bag', 'rdf:seq', 'rdf:alt', 
                                                 'rdf:rdf', 'rdf:description', 'xmp:bag', 
                                                 'xmp:seq', 'xmp:alt', 'rdf:parse_type'}
                            and not tag.startswith(('rdf:', 'xmp:', 'xml:', 'dc:'))
                            and len(tag) > 1  # Remove single character tags
                        ]
                        
                        # Remove any duplicates while preserving order
                        seen = set()
                        metadata.tags = [tag for tag in metadata.tags if not (tag in seen or seen.add(tag))]
                        
                        # If we somehow ended up with no valid tags, use a default
                        if not metadata.tags:
                            metadata.tags = ["untagged"]
                    
                    # Parse dominant colors if available
                    if 'xmpDM:ColorantSwatchList' in xmp_content:
                        colors_section = xmp_content.split('xmpDM:ColorantSwatchList', 1)[1].split('</xmpDM:ColorantSwatchList>', 1)[0]
                        color_swatches = re.findall(r'<rdf:li[^>]*>.*?</rdf:li>', colors_section, re.DOTALL)
                        
                        for swatch in color_swatches:
                            try:
                                # Extract color name and percentage using regex
                                name_match = re.search(r'xmpDM:swatchName="([^"]+)"', swatch)
                                hex_match = re.search(r'xmpDM:value="(#[0-9A-Fa-f]{6})"', swatch)
                                
                                if not hex_match:
                                    continue
                                    
                                hex_color = hex_match.group(1).upper()
                                color_name = name_match.group(1) if name_match else self._find_closest_color(hex_color)
                                
                                # Convert hex to RGB
                                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
                                
                                # Convert to HSL
                                h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
                                h = round(h * 360, 1)
                                s = round(s * 100, 1)
                                l = round(l * 100, 1)
                                
                                # Add color to metadata
                                color_info = {
                                    'hex': hex_color,
                                    'rgb': (r, g, b),
                                    'hsl': (h, s, l),
                                    'percentage': 0.0,  # Will be updated later if available
                                    'name': color_name,
                                    'temperature': 'warm' if h < 60 or h > 300 else 'cool',
                                    'shade': self._get_color_shade((r, g, b), h, l, s)
                                }
                                
                                # Try to get percentage if available
                                percent_match = re.search(r'\((\d+\.?\d*)%\)', swatch)
                                if percent_match:
                                    try:
                                        color_info['percentage'] = float(percent_match.group(1)) / 100.0
                                    except (ValueError, IndexError):
                                        pass
                                        
                                metadata.dominant_colors.append(color_info)
                                
                            except Exception as e:
                                print(f"Error parsing color swatch: {e}")
                    
                    # Sort colors by percentage (highest first)
                    if metadata.dominant_colors:
                        metadata.dominant_colors.sort(key=lambda x: x.get('percentage', 0), reverse=True)
                    
                except Exception as e:
                    print(f"Error reading XMP file: {e}")
            
            # If no XMP or no colors found, try to get colors directly from image
            if not metadata.dominant_colors:
                try:
                    metadata.dominant_colors = self.get_dominant_colors(img)
                except Exception as e:
                    print(f"Error getting dominant colors: {e}")
            
            return metadata
            
        except Exception as e:
            print(f"Error reading metadata: {e}")
            # Return minimal metadata with error information
            return ImageMetadata(
                path=image_path,
                caption=f"Error: {str(e)}",
                tags=["error"],
                width=0,
                height=0,
                format="",
                size_mb=0.0,
                dominant_colors=[]
            )
        finally:
            if 'img' in locals():
                img.close()

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image and return its metadata with tags and caption."""
        try:
            print(f"🔍 DEBUG: Starting to process image: {image_path}")
            
            # Preprocess the image
            img = self.preprocess_image(image_path)
            if img is None:
                print(f"❌ DEBUG: Failed to preprocess image: {image_path}")
                return {
                    "error": "Failed to load image",
                    "path": image_path,
                    "caption": Path(image_path).stem.replace('_', ' ').title(),
                    "tags": ["error"]
                }
            
            print(f"✅ DEBUG: Image preprocessed successfully. Size: {img.size}, Mode: {img.mode}")
            
            # Get existing metadata
            metadata = self.read_metadata(image_path)
            print(f"📄 DEBUG: Existing metadata - Caption: '{metadata.caption}', Tags: {metadata.tags}")
            
            # Always generate AI content for better results (or check if we need to regenerate)
            should_generate_ai = True  # Force AI generation for now
            
            # Alternative: Only skip AI if we have high-quality existing data
            # should_generate_ai = (
            #     not metadata.tags or 
            #     not metadata.caption or 
            #     metadata.tags == ["untagged"] or
            #     metadata.caption == "Untitled" or
            #     len(metadata.tags) < 3  # Regenerate if we have fewer than 3 tags
            # )
            
            if should_generate_ai:
                print(f"🤖 DEBUG: Generating AI content (forced regeneration enabled)")
                self.load_models()
                print(f"✅ DEBUG: Models loaded. CLIP: {self.clip_model is not None}, BLIP: {self.blip_model is not None}")
                
                # Generate caption
                try:
                    print(f"📝 DEBUG: Generating caption...")
                    metadata.caption = self.generate_caption(img)
                    print(f"✅ DEBUG: Caption generated: '{metadata.caption}'")
                except Exception as e:
                    print(f"❌ DEBUG: Error generating caption: {e}")
                    import traceback
                    traceback.print_exc()
                    metadata.caption = Path(image_path).stem.replace('_', ' ').title()
                
                # Get tags
                try:
                    print(f"🏷️ DEBUG: Generating tags...")
                    tags = self.get_tags(img)
                    print(f"✅ DEBUG: Raw tags generated: {len(tags)} tags")
                    metadata.tags = [t["tag"] for t in tags[:5]]  # Top 5 tags
                    metadata.clip_scores = {t["tag"]: float(t["score"]) for t in tags}  # Convert numpy types
                    print(f"✅ DEBUG: Final tags: {metadata.tags}")
                except Exception as e:
                    print(f"❌ DEBUG: Error generating tags: {e}")
                    import traceback
                    traceback.print_exc()
                    metadata.tags = ["untagged"]
                    metadata.clip_scores = {}
            else:
                print(f"ℹ️ DEBUG: Using existing metadata - no AI generation needed")
            
            # Get dominant colors if needed
            if not metadata.dominant_colors:
                try:
                    print(f"🎨 DEBUG: Extracting dominant colors...")
                    metadata.dominant_colors = self.get_dominant_colors(img)
                    print(f"✅ DEBUG: Colors extracted: {len(metadata.dominant_colors)} colors")
                except Exception as e:
                    print(f"❌ DEBUG: Error getting dominant colors: {e}")
                    metadata.dominant_colors = []
            
            # Convert colors to serializable format
            serializable_colors = []
            for color in metadata.dominant_colors:
                try:
                    if isinstance(color, dict):
                        # Ensure RGB and HSL values are lists of native Python types
                        rgb = color.get('rgb', (0, 0, 0))
                        if hasattr(rgb, 'tolist'):  # Convert numpy arrays to lists
                            rgb = rgb.tolist()
                            
                        hsl = color.get('hsl', (0, 0, 0))
                        if hasattr(hsl, 'tolist'):  # Convert numpy arrays to lists
                            hsl = hsl.tolist()
                            
                        serializable_colors.append({
                            'hex': str(color.get('hex', '')),
                            'rgb': {
                                'r': int(rgb[0]) if len(rgb) > 0 else 0,
                                'g': int(rgb[1]) if len(rgb) > 1 else 0,
                                'b': int(rgb[2]) if len(rgb) > 2 else 0,
                            },
                            'hsl': {
                                'h': float(hsl[0]) if len(hsl) > 0 else 0.0,
                                's': float(hsl[1]) if len(hsl) > 1 else 0.0,
                                'l': float(hsl[2]) if len(hsl) > 2 else 0.0,
                            },
                            'name': str(color.get('name', '')),
                            'percentage': float(color.get('percentage', 0.0)),
                            'temperature': str(color.get('temperature', 'neutral')),
                            'shade': str(color.get('shade', '')),
                        })
                except Exception as e:
                    print(f"❌ DEBUG: Error serializing color {color}: {e}")
            
            # Update metadata with serialized colors
            metadata.dominant_colors = serializable_colors
            
            # Write metadata back to file
            print(f"💾 DEBUG: Writing metadata to file...")
            success, _ = self.write_metadata_to_image(image_path, metadata)
            if not success:
                print(f"⚠️ DEBUG: Failed to write metadata for {image_path}")
            else:
                print(f"✅ DEBUG: Metadata written successfully")
            
            # Prepare response
            response = {
                "path": str(Path(image_path).name),
                "caption": metadata.caption or Path(image_path).stem.replace('_', ' ').title(),
                "tags": metadata.tags or ["untagged"],
                "all_tags": [
                    {"tag": t, "score": float(s), "percentage": round(float(s) * 100, 1)}
                    for t, s in (metadata.clip_scores or {}).items()
                ][:10],
                "colors": metadata.dominant_colors or [],
                "dimensions": f"{metadata.width}x{metadata.height}" if metadata.width and metadata.height else "",
                "size_mb": metadata.size_mb or round(os.path.getsize(image_path) / (1024 * 1024), 2),
                "format": metadata.format or Path(image_path).suffix[1:].upper(),
            }
            
            print(f"🎯 DEBUG: Final response prepared with {len(response['tags'])} tags and caption: '{response['caption']}'")
            return response
            
        except Exception as e:
            print(f"💥 DEBUG: Critical error processing image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "path": image_path,
                "caption": Path(image_path).stem.replace('_', ' ').title(),
                "tags": ["error"]
            }

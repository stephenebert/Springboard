from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Local COCO image (replace with any valid file)
image_path = "/Users/steph/Library/CloudStorage/OneDrive-Personal/Desktop/Springboard/Springboard/Capstone/step2/data/coco/train2017/000000000009.jpg"
image = Image.open(image_path).convert("RGB")

# Generate caption
inputs = processor(image, return_tensors="pt")
out_ids = model.generate(**inputs)
caption = processor.decode(out_ids[0], skip_special_tokens=True)

print("Generated caption:", caption)

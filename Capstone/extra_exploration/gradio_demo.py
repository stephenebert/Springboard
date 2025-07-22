import time, faiss, gradio as gr, torch, numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration, logging as hf_log
hf_log.set_verbosity_error()

print("🟢 fresh run", time.strftime("%H:%M:%S"))

FAISS_INDEX   = "scripts/coco_caption_clip.index"
CAPTION_ARRAY = "scripts/coco_caption_texts.npy"

# Test basic FAISS functionality first
print("Testing basic FAISS functionality...")
try:
    test_index = faiss.IndexFlatL2(512)
    test_vec = np.random.random((1, 512)).astype(np.float32)
    test_vec = np.ascontiguousarray(test_vec)
    test_index.add(test_vec)
    D, I = test_index.search(test_vec, 1)
    print(f"Basic FAISS test passed: D={D[0][0]:.3f}, I={I[0][0]}")
    FAISS_WORKING = True
except Exception as e:
    print(f"Basic FAISS test failed: {e}")
    FAISS_WORKING = False

device     = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load models
try:
    blip_proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base").to(device).eval()
    clip_model = SentenceTransformer("clip-ViT-B-32")
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Load FAISS index and captions
try:
    if FAISS_WORKING:
        index = faiss.read_index(FAISS_INDEX)
        captions = np.load(CAPTION_ARRAY, allow_pickle=True)
        print(f"FAISS index loaded: {index.ntotal} vectors, dimension {index.d}")
    else:
        print("FAISS not working, will use fallback similarity search")
        index = None
        captions = np.load(CAPTION_ARRAY, allow_pickle=True)
        # Create embeddings for all captions for fallback
        print("Creating embeddings for fallback search...")
        caption_embeddings = clip_model.encode(captions.tolist(), normalize_embeddings=True, convert_to_numpy=True)
        caption_embeddings = np.array(caption_embeddings, dtype=np.float32)
        print(f"Created {len(caption_embeddings)} caption embeddings")
except Exception as e:
    print(f"Error loading FAISS index or captions: {e}")
    raise

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor for BLIP model"""
    # Convert to RGB and resize
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize((384, 384), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Apply BLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img_normalized = (img_array - mean) / std
    
    # Convert to tensor format [1, 3, H, W]
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
    return img_tensor.unsqueeze(0).to(device)

def fallback_similarity_search(query_vec, k=5):
    """Fallback similarity search using numpy when FAISS fails"""
    # Compute cosine similarity
    similarities = np.dot(caption_embeddings, query_vec.T).flatten()
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    
    # Return in FAISS format (distances, indices)
    distances = 1 - similarities[top_indices]  # Convert similarity to distance
    return distances.reshape(1, -1), top_indices.reshape(1, -1)

def safe_faiss_search(vec, k=5):
    """Safely perform FAISS search with multiple fallback methods"""
    if not FAISS_WORKING or index is None:
        return fallback_similarity_search(vec, k)
    
    # Try multiple vector preparation methods
    methods = [
        lambda v: v,  # Use as-is
        lambda v: np.ascontiguousarray(v),  # Ensure contiguous
        lambda v: np.array(v, dtype=np.float32, copy=True),  # Force copy
        lambda v: np.array(v.tolist(), dtype=np.float32),  # Convert via list
    ]
    
    for i, method in enumerate(methods):
        try:
            vec_processed = method(vec)
            if vec_processed.ndim == 1:
                vec_processed = vec_processed.reshape(1, -1)
            
            # Verify array properties
            if not vec_processed.flags.c_contiguous:
                vec_processed = np.ascontiguousarray(vec_processed)
            
            print(f"Method {i+1}: shape={vec_processed.shape}, dtype={vec_processed.dtype}, contiguous={vec_processed.flags.c_contiguous}")
            
            D, I = index.search(vec_processed, k)
            print(f"FAISS search successful with method {i+1}")
            return D, I
            
        except Exception as e:
            print(f"Method {i+1} failed: {e}")
            continue
    
    # If all FAISS methods fail, use fallback
    print("⚠️  All FAISS methods failed, using fallback similarity search")
    return fallback_similarity_search(vec, k)

@torch.inference_mode()
def retrieve(img: Image.Image, k: int = 5):
    """Main retrieval function"""
    try:
        if img is None:
            return "No image provided", "Please upload an image."
        
        # Ensure k is within bounds
        k = min(k, len(captions))
        
        print(f"Processing image with k={k}")
        
        # Generate caption with BLIP
        px = pil_to_tensor(img)
        ids = blip_model.generate(px, max_new_tokens=20)
        blip_cap = blip_proc.tokenizer.decode(ids[0], skip_special_tokens=True)
        print(f"BLIP caption: {blip_cap}")

        # Get embeddings from CLIP model
        embeddings = clip_model.encode([blip_cap], normalize_embeddings=True, convert_to_numpy=True)
        
        # Ensure proper numpy array format
        vec = np.array(embeddings, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        
        print(f"Embedding shape: {vec.shape}, dtype: {vec.dtype}")
        
        # Perform similarity search
        D, I = safe_faiss_search(vec, k)
        
        # Format results
        if FAISS_WORKING and index is not None:
            neigh = [f"**{i+1}.** *distance {D[0][i]:.3f}*<br>{captions[I[0][i]]}" 
                     for i in range(k)]
        else:
            neigh = [f"**{i+1}.** *distance {D[0][i]:.3f}*<br>{captions[I[0][i]]}" 
                     for i in range(k)]
        
        return blip_cap, "<br><br>".join(neigh)
        
    except Exception as e:
        print(f"Error in retrieve: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "Please try again with a different image."

# Create Gradio interface
demo = gr.Interface(
    fn=retrieve,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(1, 10, 5, 1, label="Number of Similar Captions")
    ],
    outputs=[
        gr.Textbox(label="BLIP Generated Caption"),
        gr.HTML(label="Most Similar COCO Captions")
    ],
    title="Image-to-Text Retrieval Demo (BLIP + CLIP + FAISS)",
    description=("Upload an image → AI generates caption (BLIP) → finds embedding (CLIP) → "
                 "retrieves most similar captions from COCO dataset" + 
                 (" (FAISS)" if FAISS_WORKING else " (Fallback Search)"))
)

if __name__ == "__main__":
    print("Launching Gradio demo...")
    demo.launch(share = True)  # add share=True if you need a public link

"""
Usage:
conda activate capstone-gradio-py310
cd ~/Desktop/Springboard/Capstone/extra_credit
python gradio_demo.py
"""
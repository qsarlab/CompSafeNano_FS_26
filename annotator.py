import gradio as gr
import numpy as np
import cv2
import torch
import os
import zipfile
import tempfile
from ultralytics import SAM
from ultralytics.models.sam import SAM3SemanticPredictor
import time


# 1. Initialize the Models
base_model = SAM("sam3.pt")
base_model.to("mps")
base_model.model.half() 

text_overrides = dict(
    conf=0.5, 
    iou=0.4, 
    retina_masks=True, 
    task="segment", 
    mode="predict", 
    model="sam3.pt", 
    device="mps", 
    half=True
)
text_predictor = SAM3SemanticPredictor(overrides=text_overrides)

# --- Helper Functions ---

def extract_masks_to_numpy(ultralytics_result):
    masks = []
    if ultralytics_result and ultralytics_result[0].masks is not None:
        for m in ultralytics_result[0].masks.data:
            masks.append(m.cpu().numpy().astype(bool))
    return masks

def overlay_masks(image_path, masks):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if not masks:
        return img
        
    overlay = img.copy()
    color = np.array([255, 50, 50], dtype=np.uint8) 
    
    for mask in masks:
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0])) > 0
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
        
    return overlay

def get_clean_image(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Core App Logic (Fixed State Management) ---

def process_batch(paths, prompt_text, conf_val, iou_val, progress=gr.Progress()):
    """Core function to run the batch. Creates a BRAND NEW state dict."""
    # CRITICAL FIX 1: We create a fresh dictionary instead of clearing the old one
    new_state = {} 
    
    if not paths: return paths, 0, None, None, "No images loaded.", new_state
    
    # Update the predictor's internal arguments
    text_predictor.args.conf = conf_val
    text_predictor.args.iou = iou_val
    
    for current_path in progress.tqdm(paths, desc="🧠 AI Batch Segmenting..."):
        text_predictor.set_image(current_path)
        results = text_predictor(text=[prompt_text.strip()], conf=conf_val, iou=iou_val, verbose=False) 
        new_state[current_path] = extract_masks_to_numpy(results)
            
    first_path = paths[0]
    orig_img = get_clean_image(first_path)
    active_masks = new_state[first_path]
    rendered_img = overlay_masks(first_path, active_masks)
    
    status = f"✅ Batch complete! Image 1 of {len(paths)} | Found {len(active_masks)} particles."
    
    # Return the new_state so Gradio registers the update
    return paths, 0, orig_img, rendered_img, status, new_state

def load_files(files, prompt_text, conf_val, iou_val):
    paths = [f.name for f in files] if files else []
    return process_batch(paths, prompt_text, conf_val, iou_val)

def rerun_files(paths, prompt_text, conf_val, iou_val):
    return process_batch(paths, prompt_text, conf_val, iou_val)

def change_image(step, image_paths, index, state_dict):
    """Navigation doesn't modify state, just reads it."""
    if not image_paths: return 0, None, None, "No images loaded.", state_dict
    
    new_index = max(0, min(len(image_paths) - 1, index + step))
    current_path = image_paths[new_index]
    
    orig_img = get_clean_image(current_path)
    active_masks = state_dict.get(current_path, [])
    rendered_img = overlay_masks(current_path, active_masks)
    
    status = f"Image {new_index + 1} of {len(image_paths)} | Count: {len(active_masks)} particles."
    
    return new_index, orig_img, rendered_img, status, state_dict

def handle_click(image_paths, index, state_dict, tool_mode, evt: gr.SelectData):
    """Handles manual clicks and forces Gradio to save the updated state."""
    if not image_paths: return None, None, "No image.", state_dict
    
    # CRITICAL FIX 2: Copy the state dictionary so Gradio knows it changed!
    new_state = state_dict.copy()
    
    current_path = image_paths[index]
    orig_img = get_clean_image(current_path)
    
    # Copy the specific list of masks for this image
    active_masks = new_state.get(current_path, []).copy()
    click_x, click_y = evt.index[0], evt.index[1]
    
    if tool_mode == "Add Particle":
        results = base_model.predict(
            source=current_path,
            device="mps",
            points=[[click_x, click_y]],
            labels=[1],
            retina_masks=True,
            verbose=False
        )
        new_masks = extract_masks_to_numpy(results)
        if new_masks:
            active_masks.extend(new_masks)
            
    elif tool_mode == "Remove Artifact":
        kept_masks = []
        for mask in active_masks:
            if click_y < mask.shape[0] and click_x < mask.shape[1]:
                if mask[click_y, click_x]: 
                    continue 
            kept_masks.append(mask)
        active_masks = kept_masks

    # Save the updated masks back into our fresh state dictionary
    new_state[current_path] = active_masks
    
    rendered_img = overlay_masks(current_path, active_masks)
    status = f"Image {index + 1} of {len(image_paths)} | Count updated to {len(active_masks)}."
    
    # Return the newly updated dictionary
    return orig_img, rendered_img, status, new_state

def export_results(image_paths, state_dict):
    """Packages the truly final, updated masks into a uniquely named ZIP."""
    if not image_paths or not state_dict:
        # If no images, just return the button without a file
        return gr.DownloadButton(value=None)
        
    temp_dir = tempfile.mkdtemp()
    
    # CRITICAL FIX 1: Add a timestamp to bypass browser caching
    timestamp = int(time.time())
    zip_filename = f"nanoparticle_masks_{timestamp}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for path in image_paths:
            if path in state_dict:
                active_masks = state_dict[path]
                base_name = os.path.basename(path)
                name, _ = os.path.splitext(base_name)
                
                if len(active_masks) > 0:
                    shape = active_masks[0].shape
                else:
                    img = cv2.imread(path)
                    shape = img.shape[:2]

                combined_mask = np.zeros(shape, dtype=np.uint8)
                
                for m in active_masks:
                    if m.shape != shape:
                        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0])) > 0
                    combined_mask = np.logical_or(combined_mask, m)

                mask_img = (combined_mask * 255).astype(np.uint8)
                mask_name = f"{name}_mask.png"
                mask_path = os.path.join(temp_dir, mask_name)
                cv2.imwrite(mask_path, mask_img)
                zipf.write(mask_path, arcname=mask_name)
                
    # CRITICAL FIX 2: Explicitly update the Gradio component with the new file
    return gr.DownloadButton(value=zip_path)

# --- Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.Markdown("# 🔬 Nano Annotator")
    
    image_paths = gr.State([])
    current_idx = gr.State(0)
    state_dict = gr.State({}) 
    
    with gr.Row():
        # Left Column: Upload, Parameters, and Export
        with gr.Column(scale=1):
            file_upload = gr.File(file_count="multiple", label="Upload Image Folder (Max 30)")
            
            with gr.Accordion("⚙️ Change Parameters", open=False):
                param_text = gr.Textbox(value="dots", label="Text Prompt")
                param_conf = gr.Slider(minimum=0.05, maximum=1.0, value=0.5, step=0.05, label="Confidence (conf)")
                param_iou = gr.Slider(minimum=0.05, maximum=1.0, value=0.4, step=0.05, label="IoU (Overlap Merge)")
                btn_rerun = gr.Button("🔄 Run with New Parameters", variant="secondary")
            
            
            
        # Right Column: Visualizer and Contextual Controls
        with gr.Column(scale=4):
            with gr.Row():
                # Sub-column 1: Original Image and Annotation Tools
                with gr.Column():
                    original_img = gr.Image(type="numpy", label="Original Reference", interactive=False)
                    
                    gr.Markdown("### Annotation Tools")
                    tool_mode = gr.Radio(
                        choices=["Add Particle", "Remove Artifact"], 
                        value="Add Particle", 
                        label="Click Action"
                    )
                    
                # Sub-column 2: Annotated Canvas and Navigation
                with gr.Column():
                    display_img = gr.Image(type="numpy", label="Annotated Canvas (Click to Edit)", interactive=True)
                    
                    gr.Markdown("### Navigation")
                    with gr.Row():
                        btn_prev = gr.Button("⬅️ Previous")
                        btn_next = gr.Button("Next ➡️")
                        
                    # Moving the status text here makes sense so you can see counts as you navigate
                    status_text = gr.Textbox(label="Status", interactive=False)
                    gr.Markdown("### Export")
                    btn_export = gr.DownloadButton("💾 Download Pure Masks (ZIP)", variant="primary")

    # Events
    file_upload.upload(
        fn=load_files, 
        inputs=[file_upload, param_text, param_conf, param_iou], 
        outputs=[image_paths, current_idx, original_img, display_img, status_text, state_dict]
    )
    
    btn_rerun.click(
        fn=rerun_files,
        inputs=[image_paths, param_text, param_conf, param_iou],
        outputs=[image_paths, current_idx, original_img, display_img, status_text, state_dict]
    )
    
    display_img.select(
        fn=handle_click,
        inputs=[image_paths, current_idx, state_dict, tool_mode],
        outputs=[original_img, display_img, status_text, state_dict]
    )
    
    btn_prev.click(
        fn=lambda paths, idx, state: change_image(-1, paths, idx, state),
        inputs=[image_paths, current_idx, state_dict],
        outputs=[current_idx, original_img, display_img, status_text, state_dict]
    )
    
    btn_next.click(
        fn=lambda paths, idx, state: change_image(1, paths, idx, state),
        inputs=[image_paths, current_idx, state_dict],
        outputs=[current_idx, original_img, display_img, status_text, state_dict]
    )
    
    btn_export.click(
        fn=export_results,
        inputs=[image_paths, state_dict],
        outputs=btn_export
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)
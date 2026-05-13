import os
import csv
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor

# 1. Setup Configuration for M1
overrides = dict(
    conf=0.8,          
    iou=0.4,            
    retina_masks=True,  
    task="segment",
    mode="predict",
    model="sam3.pt",
    device="mps",
    half=True
)

predictor = SAM3SemanticPredictor(overrides=overrides)

# 2. Define your batch settings
input_folder = Path("samples")
# Add as many text variations here as you want to test
text_prompts = ["dots"] 
csv_filename = "nanoparticle_conf_80_counts.csv"

# Ensure the input folder exists to prevent crashes
if not input_folder.exists():
    print(f"❌ Error: Could not find the folder '{input_folder}'")
    exit()

# 3. Create output folders for each text prompt
for prompt in text_prompts:
    # Replace spaces with underscores for clean folder names (e.g., "results_small_circular_grains")
    folder_name = f"results_conf_80_{prompt.replace(' ', '_')}"
    Path(folder_name).mkdir(parents=True, exist_ok=True)

# List to hold our data for the CSV
results_data = [["Image_Name", "Text_Prompt", "Particle_Count"]]

# 4. Batch Processing Loop
# This grabs all PNG and JPG files in your samples folder
valid_extensions = {".png", ".jpg", ".jpeg"}
image_files = [f for f in input_folder.iterdir() if f.suffix.lower() in valid_extensions]

print(f"Found {len(image_files)} images to process. Starting batch run...\n")

for image_file in image_files:
    print(f"⚙️ Processing {image_file.name}...")
    
    # Calculate the heavy image embedding ONCE per image
    predictor.set_image(str(image_file))
    
    for prompt in text_prompts:
        sanitized_prompt = prompt.replace(' ', '_')
        
        # Run the lightweight text decoder
        results = predictor(text=[prompt])
        count = 0
        
        # Check for detections
        if results and results[0].masks is not None:
            count = len(results[0].masks)
            
            # Construct the save path: results_dots/11500X101974_dots.jpg
            save_name = f"{image_file.stem}_{sanitized_prompt}.jpg"
            save_path = str(Path(f"results_conf_80_{sanitized_prompt}") / save_name)
            
            # Save the visual result
            results[0].save(filename=save_path)
            
        # Append data for the CSV
        results_data.append([image_file.name, prompt, count])

# 5. Export the CSV File
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(results_data)

print(f"\n✅ Batch processing complete! Results saved to folders and '{csv_filename}'.")
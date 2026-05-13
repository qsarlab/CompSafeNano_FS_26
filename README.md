🔬 Nano Annotator
A Human-in-the-Loop (HITL) interactive annotation dashboard designed for the rapid, high-precision segmentation of scientific imagery. Built specifically for the DOPAD nanoparticle dataset, this tool wraps the cutting-edge Segment Anything Model 3.1 (SAM 3.1) in a user-friendly Gradio interface.

This application is heavily optimized for Apple Silicon (M1/M2/M3), utilizing Metal Performance Shaders (MPS) and Half-Precision (FP16) inference to run the massive 3.45GB SAM 3-Large model locally and smoothly on machines with 16GB of unified memory.

✨ Key Features
🧠 Zero-Shot Concept Segmentation: Uses SAM 3.1's vision-language capabilities to generate baseline masks using simple text prompts (e.g., "dots" or "circular grains").

⚡ Apple Silicon Optimization: Bypasses CUDA dependencies and runs natively on Mac GPUs via device="mps" and .half() precision, dropping inference times to sub-second speeds.

🎛️ Dynamic Parameter Tuning: Adjust AI Confidence and Intersection-over-Union (IoU) thresholds on the fly to aggressively filter out background noise or prevent over-segmentation.

🖱️ Interactive Point-Prompting: A side-by-side workspace allows you to manually correct the AI baseline. Click the canvas to Add overlooked nanoparticles or Remove generated artifacts.

📦 Batch Processing: Upload a folder of up to 30 images at once. The app front-loads the heavy processing, allowing for zero-latency navigation between images during the manual review phase.

💾 Training-Ready Export: Exports a ZIP archive containing pure, lossless binary .png masks, perfectly formatted for training downstream computer vision models.

🛠️ Prerequisites & Installation
1. Hardware Requirements
OS: macOS (Optimized for Apple Silicon M-series chips).

RAM: 16GB minimum (Required to load the sam3.pt Large model in FP16).

2. Environment Setup
It is highly recommended to use a dedicated Python environment (Conda or venv):

Bash
conda create -n sam3 python=3.11
conda activate sam3
3. Install Dependencies
Install the required Python libraries. Note: We use Ultralytics as it natively handles SAM 3 architecture without requiring complex manual installations of Facebook's research code.

Bash
pip install torch torchvision torchaudio
pip install ultralytics gradio opencv-python numpy
4. Download Model Weights
Because SAM 3.1 requires a Meta AI license agreement, the weights must be downloaded manually:

Go to the Meta SAM 3 Hugging Face Repository.

Accept the license agreement.

Download the sam3.pt file.

Place the sam3.pt file directly into the same root folder as your annotator.py script.

🚀 Usage
Open your terminal and navigate to the project directory.

Launch the application:

Bash
python annotator.py
The app will initialize the model in your M1 GPU. Once ready, it will automatically open a tab in your default web browser (usually at http://127.0.0.1:7860).

The Annotation Workflow
Upload: Drag and drop a batch of electron microscopy images (e.g., .png, .jpg) into the upload box.

AI Baseline: The app will automatically run the default text prompt ("dots") across all images in the background.

Review & Tune: If the baseline is missing particles, open the ⚙️ Change AI Parameters accordion, lower the confidence slider, and click Run with New Parameters.

Manual Correction: * Select Add Particle and left-click an unmasked nanoparticle on the canvas to segment it.

Select Remove Artifact and left-click a red mask to delete it.

Export: Click Download Pure Masks (ZIP). The app will generate a unique ZIP file containing solid black-and-white .png masks for every image you reviewed.

📁 Output Format
The export function generates binary masks designed for deep learning frameworks (like PyTorch or TensorFlow).

Background: Black (Pixel value 0)

Nanoparticles: White (Pixel value 255)

Format: Lossless .png to prevent JPEG compression artifacts around the mask edges.

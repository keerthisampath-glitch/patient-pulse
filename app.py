"""
PatientPulse AI — Hugging Face Space Production Entrypoint
Runs on Hugging Face Free Tier (Gradio SDK — 16 GB RAM, 2 vCPU)
"""

import os
import sys
import uvicorn

# Ensure project directories are in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "App"))

from App.backend.main import app

# Hugging Face Gradio Space compatibility
try:
    import gradio as gr
    with gr.Blocks(title="PatientPulse AI Clinical Platform") as demo:
        gr.Markdown("# PatientPulse AI Clinical Platform\nAccess the full clinical web portal at the root URL `/`.")
    app = gr.mount_gradio_app(app, demo, path="/hf-check")
except ImportError:
    pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[Hugging Face] Launching PatientPulse AI on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

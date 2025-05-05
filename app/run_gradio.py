import logging
import gradio as gr
import os
import time
from core.llm_interface import creative_engine
from core.db_manager import db_manager
from pathlib import Path
import shutil
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define base directory relative to this script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "datastore", "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_MODELS_DIR = os.path.join(STATIC_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
HTML_TEMPLATE_PATH = os.path.join(ASSETS_DIR, "model_viewer_template.html")

HTML_HEIGHT = 400 # Define height for the model viewer iframe

# Add these constants near the top of the file
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_cache")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
HTML_TEMPLATE_PATH = os.path.join(ASSETS_DIR, "model_viewer_template.html")
HTML_HEIGHT = 660
HTML_WIDTH = 790

def build_model_viewer_html(model_path, height=HTML_HEIGHT, width=HTML_WIDTH, textured=False):
    """Generate HTML for model viewer with proper static file paths"""
    if not model_path or not os.path.exists(model_path):
        return "<p>Error: Model file not found.</p>"

    try:
        # Create a unique subfolder in cache directory
        save_folder = os.path.join(CACHE_DIR, str(time.time()))
        os.makedirs(save_folder, exist_ok=True)

        # Copy model file to the cache folder
        model_filename = os.path.basename(model_path)
        cache_model_path = os.path.join(save_folder, model_filename)
        shutil.copy2(model_path, cache_model_path)

        # Create relative path for static serving
        rel_model_path = f"./{model_filename}"
        
        # Define the HTML template string directly
        template_html = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>3D Model Viewer</title>
                <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
                <style>
                    body {
                        margin: 0;
                        background: #181818; /* Darker background for contrast */
                    }
                    model-viewer {
                        width: 100%;
                        height: #height#px;
                        background-color: #181818; /* Darker background for model */
                        border-radius: 8px;
                        box-shadow: 0 2px 16px rgba(0,0,0,0.12);
                    }
                </style>
            </head>
            <body>
                <model-viewer 
                    src="#src#" 
                    alt="3D Model" 
                    camera-controls 
                    auto-rotate 
                    ar 
                    shadow-intensity="1"
                    exposure="0.7"
                    environment-image="neutral"
                    tone-mapping="aces"
                    shadow-softness="1"
                    style="background-color: #181818;"
                >
                </model-viewer>
                <div style="color:#888; font-size:12px; text-align:center; margin-top:8px;">
                    <em>Note: Model color is determined by the 3D file. For a dark model, upload or generate a model with dark materials.</em>
                </div>
            </body>
        </html>
        """

        # Replace placeholders
        viewer_html = template_html.replace('#height#', str(height))
        viewer_html = viewer_html.replace('#src#', rel_model_path)

        # Save the customized viewer HTML to the cache directory
        viewer_html_path = os.path.join(save_folder, 'viewer.html')
        with open(viewer_html_path, 'w', encoding='utf-8') as f:
            f.write(viewer_html)

        # Create iframe URL using relative path from CACHE_DIR
        rel_path = os.path.relpath(viewer_html_path, CACHE_DIR)
        iframe_url = f"/static/{rel_path}"
        
        return f"""
            <div style='height: {height}px; width: 100%;'>
                <iframe src="{iframe_url}" height="{height}" width="100%" frameborder="0"></iframe>
            </div>
        """

    except Exception as e:
        logging.error(f"Error generating model viewer HTML: {e}")
        return f"<p>Error creating model viewer: {e}</p>"

def create_interface():
    """Create and launch the Gradio interface"""
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Setup Static Symlink ---
    os.makedirs(os.path.dirname(STATIC_MODELS_DIR), exist_ok=True)
    if os.path.exists(STATIC_MODELS_DIR) or os.path.islink(STATIC_MODELS_DIR):
        logger.info(f"Removing existing static models link/dir: {STATIC_MODELS_DIR}")
        try:
            if os.path.islink(STATIC_MODELS_DIR):
                os.unlink(STATIC_MODELS_DIR)
            else:
                # Be cautious removing directories if not expected
                logger.warning(f"Path {STATIC_MODELS_DIR} exists but is not a symlink. Manual cleanup might be needed.")
                # os.rmdir(STATIC_MODELS_DIR) # Or shutil.rmtree if it might contain files
        except OSError as e:
            logger.error(f"Error removing existing static models link/dir: {e}")

    try:
        logger.info(f"Creating symlink from {MODELS_DIR} to {STATIC_MODELS_DIR}")
        os.symlink(MODELS_DIR, STATIC_MODELS_DIR, target_is_directory=True)
    except OSError as e:
         # Handle cases where symlink creation fails (e.g., permissions, file exists)
         logger.error(f"Failed to create symlink: {e}. Ensure the target directory exists and permissions are correct.")
         # Optionally, raise the error or provide fallback behavior
         # raise e
    # --- End Static Symlink Setup ---


    def verify_model_path(model_path):
        """Verify model path exists and is supported format, return absolute path."""
        if not model_path:
            logger.warning("verify_model_path received None or empty path.")
            return None

        # Ensure path is absolute
        abs_model_path = os.path.abspath(model_path)

        if not os.path.exists(abs_model_path):
            logger.error(f"Model file not found at absolute path: {abs_model_path}")
            return None
        if not abs_model_path.lower().endswith(('.glb', '.gltf', '.obj')):
            logger.error(f"Unsupported 3D model format: {abs_model_path}")
            return None
        # Return absolute path
        return abs_model_path

    # Apply a theme for better visual appeal
    theme = gr.themes.Default(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        # Customize specific component styles if needed
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_400",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_label_text_weight="500",
        body_background_fill="#f4f7f9", # Light grey background
    )

    with gr.Blocks(theme=theme, title="Creative AI Engine") as interface:
        gr.Markdown(
            """
            # 🎨 Creative AI Engine ✨
            Transform your simple ideas into stunning visuals and interactive 3D models.
            Describe your concept, and let the AI bring it to life!
            """
        )

        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Describe Your Idea")
                    prompt_input = gr.Textbox(
                        label="Your Prompt",
                        placeholder="e.g., A futuristic robot exploring a lush alien jungle...",
                        lines=4,
                        show_label=False
                    )
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                        generate_btn = gr.Button("🚀 Generate", variant="primary", scale=2)

                with gr.Accordion("🧠 Enhanced Prompt (AI Generated)", open=True):
                     enhanced_prompt = gr.Textbox(
                        label="AI Enhanced Description",
                        lines=4,
                        interactive=False,
                        show_label=False
                    )

            # Output Column
            with gr.Column(scale=2):
                 with gr.Tabs():
                    with gr.TabItem("🖼️ Generated Image"):
                         with gr.Group():
                            image_output = gr.Image(
                                label="Generated Image",
                                type="filepath",
                                height=500, # Adjust height as needed
                                show_label=False
                            )
                    with gr.TabItem("🧊 3D Model Preview"):
                         with gr.Group():
                            # Replace gr.Model3D with gr.HTML
                            model_viewer_output = gr.HTML(
                                label="3D Model Preview",
                                min_height=510 # Match image height roughly
                            )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("📚 Creation History", open=False):
                    history = gr.DataFrame(
                        headers=["Original Prompt", "Enhanced Prompt", "Created At"],
                        datatype=["str", "str", "str"], # Use str for datetime display
                        label="Recent Generations",
                        row_count=(5, "dynamic"), # Show 5 rows, allow dynamic height
                        wrap=True
                    )
                    refresh_btn = gr.Button("🔄 Refresh History")

            with gr.Column(scale=1):
                with gr.Accordion("🔍 Search Past Creations", open=False):
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., dragon, sunset, robot...",
                        show_label=False
                    )
                    search_btn = gr.Button("🔍 Search History")
                    search_results = gr.DataFrame(
                        headers=["Original Prompt", "Enhanced Prompt", "Created At"],
                        datatype=["str", "str", "str"], # Use str for datetime display
                        label="Search Results",
                        row_count=(5, "dynamic"),
                        wrap=True
                    )


        def process_prompt(prompt):
            logger.info(f"Gradio process_prompt started for: '{prompt[:50]}...'")
            try:
                # --- Add logging before the call ---
                logger.info("Calling creative_engine.process_prompt...")
                start_call_time = time.time()
                # --- End of added block ---

                enhanced_prompt_raw, image_path, model_path = creative_engine.process_prompt(prompt)

                # --- Add logging after the call ---
                end_call_time = time.time()
                logger.info(f"creative_engine.process_prompt returned in {end_call_time - start_call_time:.2f} seconds.")
                logger.info(f"Received model path from creative_engine: {model_path}")
                # --- End of added block ---

                # Clean the enhanced prompt
                enhanced_prompt_cleaned = enhanced_prompt_raw.strip().strip('"') if enhanced_prompt_raw else ""
                logger.info(f"Cleaned enhanced prompt: '{enhanced_prompt_cleaned[:100]}...'")


                if not image_path:
                    logger.error("process_prompt: No image path received.")
                    # Return cleaned prompt even on image failure
                    return enhanced_prompt_cleaned, None, "<p>Image generation failed.</p>"

                verified_model_path = verify_model_path(model_path)
                if (verified_model_path):
                    logger.info(f"process_prompt: Verified model path: {verified_model_path}")
                    model_html = build_model_viewer_html(verified_model_path)
                    logger.info(f"process_prompt: Generated model HTML (first 100 chars): {model_html[:100]}")
                    logger.info("Gradio process_prompt finished successfully.")
                    # Return cleaned prompt
                    return enhanced_prompt_cleaned, image_path, model_html
                else:
                    logger.error(f"process_prompt: Model path verification failed for: {model_path}")
                    logger.info("Gradio process_prompt finished with image only (model verification failed).")
                    # Return cleaned prompt
                    return enhanced_prompt_cleaned, image_path, "<p>Error: Could not load 3D model (path verification failed).</p>"

            except Exception as e:
                logger.exception(f"process_prompt: Error processing prompt: {e}")
                logger.info("Gradio process_prompt finished with error.")
                # Return error message to enhanced_prompt as well for visibility
                # Ensure error message is also cleaned (though unlikely to have quotes)
                error_msg = f"Error: {e}".strip().strip('"')
                return error_msg, None, f"<p>An error occurred: {e}</p>"

        def update_history():
            generations = db_manager.get_recent_generations()
            formatted_generations = []
            if generations:
                for g in generations:
                    # Safely format the timestamp (index 5)
                    timestamp_str = ""
                    if isinstance(g[5], datetime.datetime):
                        timestamp_str = g[5].strftime('%Y-%m-%d %H:%M')
                    elif isinstance(g[5], str): # Handle if it's already a string
                        timestamp_str = g[5]
                    else: # Handle None or other unexpected types
                        timestamp_str = "N/A"
                    formatted_generations.append([str(g[1]), str(g[2]), timestamp_str])
            return formatted_generations


        def search_creations(query):
            results = db_manager.search_generations(query)
            formatted_results = []
            if results:
                 for r in results:
                    # Safely format the timestamp (index 5)
                    timestamp_str = ""
                    if isinstance(r[5], datetime.datetime):
                        timestamp_str = r[5].strftime('%Y-%m-%d %H:%M')
                    elif isinstance(r[5], str): # Handle if it's already a string
                        timestamp_str = r[5]
                    else: # Handle None or other unexpected types
                        timestamp_str = "N/A"
                    formatted_results.append([str(r[1]), str(r[2]), timestamp_str])
            return formatted_results

        def clear_outputs():
             # Clear HTML output as well
            return "", "", None, "<p style='text-align:center; color:grey;'>Outputs cleared.</p>", "", None, None # Clear prompt, enhanced, image, model, history, search

        # --- Event Wiring (update outputs list for clear) ---
        generate_btn.click(
            fn=process_prompt,
            inputs=[prompt_input],
            outputs=[enhanced_prompt, image_output, model_viewer_output]
        ).then( # Use .then to chain the history update after generation
            fn=update_history,
            inputs=[],
            outputs=[history]
        )

        clear_btn.click(
            fn=clear_outputs,
            inputs=[],
            # Ensure all relevant outputs are cleared
            outputs=[prompt_input, enhanced_prompt, image_output, model_viewer_output, search_input, history, search_results]
        )

        refresh_btn.click(
            fn=update_history,
            inputs=[],
            outputs=[history]
        )

        search_btn.click(
            fn=search_creations,
            inputs=[search_input],
            outputs=[search_results]
        )

        # Load initial history on interface load
        interface.load(update_history, None, [history])

        return interface

if __name__ == "__main__":
    logger.info("Starting Creative AI Engine...")
    # Create and configure FastAPI app
    app = FastAPI()
    
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Mount static file directory
    app.mount("/static", StaticFiles(directory=CACHE_DIR, html=True), name="static")
    
    # Create Gradio interface
    demo = create_interface()
    
    # Mount Gradio app to FastAPI
    app = gr.mount_gradio_app(app, demo, path="/")
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
import os
import tempfile
import subprocess
import gradio as gr
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import json

from ai_image_captioning.data_model.image_metadata import ImageMetadata
from ai_image_captioning.ingestion.photos_loader import GooglePhotosClient
from ai_image_captioning.agent.image_captioning import ImageCaptionAgent
from ai_image_captioning.utils.image_preprocessing import ImageTextExtractor

# Configuration
CAPTIONS_FILE = "validated_captions.parquet"
TEMP_DIR = Path(tempfile.mkdtemp())
os.makedirs("data", exist_ok=True)

# Supported multimodal models for selection
SUPPORTED_MODELS = ["llava", "llama4", "bakllava"]


class CaptionApp:
    def __init__(self):
        # Default model
        self.selected_model = SUPPORTED_MODELS[0]
        self.caption_agent = ImageCaptionAgent(model_name=self.selected_model)
        self.text_extractor = ImageTextExtractor()
        self.image_metadata = None
        self.photos_enabled = False
        self.photos_client = self._init_google_photos_client()
        self.current_photos = []
        self._init_data_store()

    def _init_google_photos_client(self) -> Optional[GooglePhotosClient]:
        """Try to load credentials from env or fallback to local file"""
        cred_json = os.environ.get("GOOGLE_PHOTOS_CREDENTIALS_JSON")
        cred_path = Path("config/credentials.json")

        if cred_json:
            try:
                parsed = json.loads(cred_json)
                temp_cred = TEMP_DIR / "creds.json"
                temp_cred.write_text(json.dumps(parsed))
                self.photos_enabled = True
                return GooglePhotosClient(str(temp_cred))
            except Exception as e:
                print(f"Invalid credentials from environment: {e}")
        elif cred_path.exists():
            self.photos_enabled = True
            return GooglePhotosClient(str(cred_path))

        print("⚠️ Google Photos credentials not found. Option disabled.")
        return None

    def _init_data_store(self):
        """Initialize the Parquet data store"""
        self.captions_path = Path("data") / CAPTIONS_FILE
        if not self.captions_path.exists():
            self.captions_df = pd.DataFrame(
                columns=[
                    "image_path",
                    "caption",
                    "validated_at",
                    "source",
                    "processing_time",
                    "created_at",
                ]
            )
            self._save_captions()
        else:
            self.captions_df = pd.read_parquet(self.captions_path)

    def _save_captions(self):
        """Save captions to Parquet file"""
        self.captions_df.to_parquet(self.captions_path)

    def _get_google_photos(self) -> List[str]:
        """Load photo metadata and return dropdown options"""
        try:
            self.current_photos = self.photos_client.get_photos()
            return [photo["filename"] for photo in self.current_photos]
        except Exception as e:
            print(f"Error loading Google Photos: {e}")
            return []

    def _get_photo_by_filename(self, filename: str) -> Optional[dict]:
        return next((p for p in self.current_photos if p["filename"] == filename), None)

    def _download_and_display_photo(self, filename: str):
        photo_item = self._get_photo_by_filename(filename)
        if not photo_item:
            return None, "Photo not found"

        try:
            temp_file = TEMP_DIR / photo_item["filename"]
            if not temp_file.exists():
                downloaded = self.photos_client.download_photo(photo_item, TEMP_DIR)
                if not downloaded:
                    return None, "Download failed"
            caption = self._generate_caption(str(temp_file))
            return str(temp_file), caption
        except Exception as e:
            return None, f"Error: {str(e)}"

    def _generate_caption(self, image_path: str) -> str:
        try:
            result = self.caption_agent.generate_caption(Path(image_path))
            return result.caption if result.success else "Caption generation failed"
        except Exception as e:
            return "Error generating caption"

    def _validate_caption(self, image_path: str, caption: str) -> str:
        try:
            if not isinstance(image_path, str):
                image_path = getattr(image_path, "name", str(image_path))

            source = "upload"
            if (
                "google_photos" in image_path.lower()
                or "contentlibrary" in image_path.lower()
            ):
                source = "google_photos"

            created_at = None
            if source == "google_photos":
                filename = Path(image_path).name
                photo_item = self._get_photo_by_filename(filename)
                if photo_item and "mediaMetadata" in photo_item:
                    created_at = photo_item["mediaMetadata"].get("creationTime")

            new_record = pd.DataFrame(
                [
                    {
                        "image_path": image_path,
                        "caption": caption,
                        "validated_at": datetime.now().isoformat(),
                        "source": source,
                        "processing_time": None,
                        "created_at": created_at,
                    }
                ]
            )

            self.captions_df = pd.concat(
                [self.captions_df, new_record], ignore_index=True
            )
            self._save_captions()
            return "✅ Caption validated and saved!"
        except Exception as e:
            return "❌ Error saving caption"

    def is_model_installed(self, model_name: str) -> bool:
        """Check if the model is installed locally using 'ollama list'."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            return model_name in result.stdout
        except Exception as e:
            print(f"Error checking model installation: {e}")
            return False

    def _on_model_change(self, model_name: str) -> str:
        """Handler for model selection changes"""
        if self.is_model_installed(model_name):
            self.selected_model = model_name
            self.caption_agent = ImageCaptionAgent(model_name=model_name)
            return f"Model '{model_name}' is selected and ready to use."
        else:
            return f"Model '{model_name}' is not installed. Please run 'ollama pull {model_name}' to install it."

    def create_interface(self):
        with gr.Blocks(title="Image Captioning App") as app:
            gr.Markdown("# 📷 AI Image Captioning")

            with gr.Row():
                with gr.Column(scale=1):
                    # Model selection dropdown
                    model_dropdown = gr.Dropdown(
                        choices=SUPPORTED_MODELS,
                        label="Select Model",
                        value=self.selected_model,
                    )
                    model_status = gr.Textbox(label="Model Status", interactive=False)

                    source_radio = gr.Radio(
                        ["Google Photos", "Manual Upload"],
                        label="Image Source",
                        value=(
                            "Google Photos" if self.photos_enabled else "Manual Upload"
                        ),
                    )
                    photo_dropdown = gr.Dropdown(
                        label="Select Photo",
                        interactive=True,
                        visible=self.photos_enabled,
                        filterable=True,
                    )
                    upload_btn = gr.UploadButton(
                        "📁 Upload Image",
                        file_types=["image"],
                        visible=not self.photos_enabled,
                    )
                    refresh_btn = gr.Button(
                        "🔄 Refresh Google Photos", visible=self.photos_enabled
                    )
                    if not self.photos_enabled:
                        gr.Markdown(
                            "⚠️ Google Photos credentials not found. Feature disabled."
                        )

                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Image Preview", interactive=False, width=900, height=600
                    )
                    caption_output = gr.Textbox(
                        label="Generated Caption", interactive=False
                    )
                    validate_btn = gr.Button("✅ Validate Caption", variant="primary")
                    validation_status = gr.Textbox(
                        label="Validation Status", interactive=False
                    )

            # Events
            model_dropdown.change(
                fn=self._on_model_change, inputs=model_dropdown, outputs=model_status
            )

            source_radio.change(
                fn=lambda x: (
                    gr.update(visible=x == "Google Photos"),
                    gr.update(visible=x == "Manual Upload"),
                    gr.update(visible=x == "Google Photos"),
                ),
                inputs=source_radio,
                outputs=[photo_dropdown, upload_btn, refresh_btn],
            )

            refresh_btn.click(
                fn=lambda: gr.update(choices=self._get_google_photos()),
                outputs=photo_dropdown,
            )

            photo_dropdown.change(
                fn=self._download_and_display_photo,
                inputs=photo_dropdown,
                outputs=[output_image, caption_output],
            )

            upload_btn.upload(
                fn=lambda x: (x.name, self._generate_caption(x.name)),
                inputs=upload_btn,
                outputs=[output_image, caption_output],
            )

            validate_btn.click(
                fn=self._validate_caption,
                inputs=[output_image, caption_output],
                outputs=validation_status,
            )

        return app


if __name__ == "__main__":
    app = CaptionApp().create_interface()
    app.launch()

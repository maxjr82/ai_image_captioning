"""
Image Captioning Agent using Ollama
"""

import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from ai_image_captioning.utils.image_preprocessing import ImageTextExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaptioningResult(BaseModel):
    """Structured result container"""

    success: bool
    caption: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class ImageCaptionAgent:
    """Fixed implementation with proper Ollama connection handling"""

    def __init__(
        self,
        model_name: str = "llava",
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 120,
        # Default Ollama port
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = base_url
        self.preprocessor = ImageTextExtractor()
        self.llm = self._initialize_model()

    def _verify_ollama_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.ConnectionError:
            return False

    def _initialize_model(self) -> ChatOllama:
        """Initialize model with connection verification"""
        if not self._verify_ollama_connection():
            raise ConnectionError(
                f"Ollama server not reachable at {self.base_url}. "
                "Please ensure Ollama is running and accessible."
            )

        try:
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                timeout=self.timeout,
                base_url=self.base_url,
                keep_alive="5m",
            )
        except Exception as e:
            err_msg = f"Model initialization failed: {str(e)}"
            logger.error(err_msg)
            raise

    def _encode_image(self, image_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Convert image to base64 with validation"""
        try:
            with Image.open(image_path) as img:
                if img.format not in ["JPEG", "PNG", "WEBP"]:
                    return None, f"Unsupported image format: {img.format}"

                buffered = BytesIO()
                img.save(buffered, format=img.format)
                return base64.b64encode(buffered.getvalue()).decode("utf-8"), None

        except UnidentifiedImageError:
            return None, "Invalid image file"
        except Exception as e:
            return None, str(e)

    def _build_messages(self, image_b64: str, ocr_result: Optional[Dict]) -> List:
        """Construct messages for the LLM, including OCR tip if available"""
        # Base system prompt
        system_prompt = (
            "You are an AI assistant that can write engaging and high-fidelity"
            " captions for any provided image.\n\n"
            "When generating captions, please follow these guidelines:\n"
            "- Concise, factual description\n"
            "- Focus on main subjects and composition\n"
            "- Mention text only if clearly visible\n"
            "- Avoid subjective interpretations\n"
            "- Do not mention that there is no visible text in the image\n"
            "- If a recognized text is provided to you, correct text errors "
            " and use it as a system-tip to help you describe the image.\n"
            "- Use maximum 50 words or 300 characters including spaces.\n"
        )
        # If OCR extracted meaningful text, add as a system-tip
        if ocr_result and ocr_result.get("text"):
            ocr_text = ocr_result["text"].strip()
            system_prompt += f"\n- Recognized text on image: '{ocr_text}'"

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    },
                    {"type": "text", "text": "Describe this image objectively."},
                ]
            ),
        ]

    def generate_caption(self, image_path: Path) -> CaptioningResult:
        """Generate caption with full error handling"""
        start_time = time.time()
        result = {"success": False}

        try:
            # Verify image exists
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Encode image
            image_b64, error = self._encode_image(image_path)
            if error:
                raise ValueError(f"Image processing failed: {error}")

            # Perform OCR extraction
            ocr_result = self.preprocessor.extract_text(image_path)

            # Prepare messages with OCR tip
            messages = self._build_messages(image_b64, ocr_result)

            # Execute with retries
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.llm.invoke(messages)
                    return CaptioningResult(
                        success=True,
                        caption=response.content.strip(),
                        processing_time=time.time() - start_time,
                    )
                except Exception as e:
                    msg = f"Attempt {attempt} failed: {str(e)}"
                    logger.warning(msg)
                    if attempt == self.max_retries:
                        raise

        except Exception as e:
            err_msg = f"Caption generation failed: {str(e)}"
            logger.error(err_msg)
            result.update(
                {"error": str(e), "processing_time": time.time() - start_time}
            )

        return CaptioningResult(**result)


if __name__ == "__main__":
    # Example usage
    agent = ImageCaptionAgent()

    test_images = [
        Path("data/test8.png"),
        Path("data/test1.jpg"),
        Path("data/test10.jpg"),
    ]

    for img_path in test_images:
        print(f"\nProcessing {img_path}:")
        result = agent.generate_caption(img_path)

        if result.success:
            print(f"Caption: {result.caption}")
            print(f"Processing time: {result.processing_time:.2f}s")
        else:
            print(f"Error: {result.error}")

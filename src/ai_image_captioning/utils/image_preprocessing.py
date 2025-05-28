import logging
import re
from pathlib import Path
from typing import Dict, Optional

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class ImageTextExtractor:
    """OCR preprocessing tool for text extraction from images with
    embedded quality checks"""

    def __init__(
        self,
        min_confidence: float = 60.0,
        min_quality: float = 0.5,
        ocr_config: str = r"--oem 3 --psm 11",
    ):
        self.min_confidence = min_confidence
        self.min_quality = min_quality
        self.ocr_config = ocr_config

        # Patterns for linguistic checks
        self.vowel_pattern = re.compile(r"[aeiouAEIOU]")
        self.consonant_pattern = re.compile(
            r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]"
        )
        self.word_pattern = re.compile(r"\b[a-zA-Z]{3,}\b")

    def _is_valid_word(self, word: str) -> bool:
        """Basic linguistic validation without dictionary"""
        if len(word) < 3:
            return False
        has_vowels = bool(self.vowel_pattern.search(word))
        has_consonants = bool(self.consonant_pattern.search(word))
        return has_vowels and has_consonants

    def _calculate_text_quality(self, text: str) -> float:
        """Evaluate text quality using basic rules"""
        clean_text = re.sub(r"[^a-zA-Z\s]", "", text).strip()
        words = re.findall(self.word_pattern, clean_text)
        if not words:
            return 0.0
        valid_words = [w for w in words if self._is_valid_word(w)]
        return len(valid_words) / len(words)

    def _preprocess_image(self, image_path: Path) -> Optional[Image.Image]:
        """Optimize image for OCR"""
        try:
            with Image.open(image_path) as img:
                img = img.convert("L")  # grayscale
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(2.0)
                img = img.point(lambda x: 0 if x < 140 else 255)
                img = img.filter(ImageFilter.MedianFilter(size=3))
                return img
        except Exception as e:
            err_msg = f"Image preprocessing failed for {image_path}: {e}"
            logger.error(err_msg)
            return None

    def _calculate_confidence(self, conf_values: list) -> float:
        """Calculate average OCR confidence score"""
        valid_confs = [float(c) for c in conf_values if c not in ("-1", None)]
        return sum(valid_confs) / len(valid_confs) if valid_confs else 0.0

    def _filter_text_results(self, ocr_data: Dict) -> Dict:
        """Filter and format OCR results"""
        filtered = {"text": "", "confidence": 0.0, "significant_text": []}
        for i, raw in enumerate(ocr_data.get("text", [])):
            text = raw.strip()
            try:
                conf = float(ocr_data["conf"][i])
            except (ValueError, TypeError):
                continue

            if text and conf >= self.min_confidence:
                filtered["text"] += f"{text} "
                filtered["confidence"] += conf
                if len(text) > 3:
                    filtered["significant_text"].append(
                        {
                            "text": text,
                            "confidence": conf,
                            "position": (
                                ocr_data["left"][i],
                                ocr_data["top"][i],
                                ocr_data["width"][i],
                                ocr_data["height"][i],
                            ),
                        }
                    )
        if filtered["significant_text"]:
            filtered["confidence"] /= len(filtered["significant_text"])
        return filtered

    def extract_text(self, image_path: Path) -> Optional[Dict]:
        """
        Extract text from image with confidence and quality scoring
        Returns None if quality or confidence below thresholds
        Returns: {
            "text": str,
            "confidence": float,
            "quality": float,
            "significant_text": list
        }
        """
        try:
            img = self._preprocess_image(image_path)
            if img is None:
                return None

            data = pytesseract.image_to_data(
                img,
                config=self.ocr_config,
                output_type=pytesseract.Output.DICT,
            )
            result = self._filter_text_results(data)

            # Calculate overall quality of extracted text
            quality = self._calculate_text_quality(result["text"])
            result["quality"] = quality

            # Only return if quality meets threshold
            if quality >= self.min_quality and result["significant_text"]:
                return result

            msg = f"Text quality too low ({quality:.2f}) or no significant"
            msg += " text found."
            logger.info(msg)
        except pytesseract.TesseractError as e:
            err_msg = f"OCR processing failed: {e}"
            logger.error(err_msg)
        except Exception as e:
            err_msg = f"Unexpected error during text extraction: {e}"
            logger.error(err_msg)
        return None

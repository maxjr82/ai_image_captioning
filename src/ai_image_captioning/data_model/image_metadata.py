import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from PIL import ExifTags, Image, UnidentifiedImageError
from pydantic import AnyUrl, BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


def _extract_exif(img: Image.Image) -> Dict[Any, Any]:
    try:
        exif_info = img._getexif()
        exif = {}
        if exif_info:
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_info.items()}
        return exif
    except Exception as e:
        warn_msg = f"Failed to extract EXIF: {e}"
        logger.warning(warn_msg)
        return {}


def _get_gps_coord(exif: Dict[Any, Any]) -> Optional[str]:
    gps_info = exif.get("GPSInfo") or {}
    if not gps_info:
        return ""
    # map numeric keys
    gps = {}
    for key, val in gps_info.items():
        name = ExifTags.GPSTAGS.get(key, key)
        gps[name] = val

    def _to_deg(vals, ref):
        try:
            d, m, s = vals
            dec = d + m / 60 + s / 3600
            dec = float(dec)
            return -dec if ref in ["S", "W"] else dec
        except Exception:
            return ""

    lat = _to_deg(
        gps.get("GPSLatitude", (0, 0, 0)), gps.get("GPSLatitudeRef", "N")
    )
    lon = _to_deg(
        gps.get("GPSLongitude", (0, 0, 0)), gps.get("GPSLongitudeRef", "E")
    )

    if lat is not None and lon is not None:
        return (lat, lon)
    return ""


def _get_location(lat: float, lon: float) -> dict:
    """
    Retrieves the city and country names for the given latitude and longitude.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        dict: A dictionary containing 'city' and 'country' keys.
    """
    try:
        geolocator = Nominatim(user_agent="image_captioning_app")
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        if location and location.raw and "address" in location.raw:
            address = location.raw["address"]
            neighbourhood = address.get("city_block", "")
            city = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("hamlet")
            )
            country = address.get("country")
            return {
                "neighbourhood": neighbourhood,
                "city": city,
                "country": country,
            }
        else:
            return {"neighbourhood": None, "city": None, "country": None}
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error: {e}")
        return {"neighbourhood": None, "city": None, "country": None}


class ImageMetadata(BaseModel):
    """
    Pydantic model for image metadata with EXIF enrichment
    """

    image_id: str = Field(
        ..., min_length=1, description="Unique identifier for the image"
    )
    filename: str = Field(
        ..., min_length=4, description="Original filename or URL"
    )
    image_location: Union[Path, AnyUrl] = Field(
        ..., description="Local path or URL of the image"
    )

    # Captions
    generated_caption: Optional[str] = Field(
        None, max_length=5000, description="AI-generated caption for the image"
    )
    validated_caption: Optional[str] = Field(
        None, max_length=5000, description="Human-validated or edited caption"
    )

    # OCR text
    extracted_text: Optional[str] = Field(
        None, max_length=10000, description="Text extracted from image via OCR"
    )

    # EXIF fields
    make: Optional[str] = Field(None, description="Camera make from EXIF")
    model: Optional[str] = Field(None, description="Camera model from EXIF")
    software: Optional[str] = Field(
        None, description="Software used to process the image"
    )
    image_width: Optional[int] = Field(
        None, description="Image width in pixels"
    )
    image_length: Optional[int] = Field(
        None, description="Image length in pixels"
    )

    # Camera settings
    aperture_value: Optional[float] = Field(
        None, description="Aperture value (FNumber) from EXIF"
    )
    focal_length: Optional[float] = Field(
        None, description="Focal length from EXIF"
    )
    brightness: Optional[float] = Field(
        None, description="Brightness from EXIF"
    )
    exposure_time: Optional[float] = Field(
        None, description="Exposure time from EXIF"
    )
    iso: Optional[int] = Field(None, description="ISO speed from EXIF")
    gps: Optional[Tuple[float, float]] = Field(
        None, description="GPS coordinates as (lat, lon)"
    )

    # Complementary fields
    location: Optional[str] = Field(
        None, description="Location name from reverse geocoding"
    )
    processed: bool = Field(
        False, description="Whether the image has been processed by a human"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when record was created or EXIF datetime",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when record was last updated",
    )

    @field_validator("image_location")
    @classmethod
    def validate_location(cls, v):
        if isinstance(v, Path) and not v.exists():
            raise ValueError(f"Path {v} does not exist")
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v):
        if v.startswith("http"):
            return v
        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        if not any(v.lower().endswith(ext) for ext in valid_ext):
            raise ValueError(f"Filename {v} has invalid extension")
        return v

    @model_validator(mode="before")
    @classmethod
    def load_exif_and_set_fields(cls, values):
        loc = values.get("image_location")
        try:
            if isinstance(loc, str):
                loc = Path(loc)
            img = Image.open(loc)
            exif = _extract_exif(img)
            # parse DateTimeOriginal
            dto = exif.get("DateTimeOriginal")
            if dto:
                try:
                    values["created_at"] = datetime.strptime(
                        dto, "%Y:%m:%d %H:%M:%S"
                    )
                except Exception:
                    pass
            values["make"] = exif.get("Make")
            values["model"] = exif.get("Model")
            values["software"] = exif.get("Software")
            values["image_width"] = exif.get("ImageWidth")
            values["image_length"] = exif.get("ImageLength")
            values["aperture_value"] = exif.get("FNumber")
            values["brightness"] = exif.get("BrightnessValue")

            fl = exif.get("FocalLength")
            if fl:
                values["focal_length"] = float(fl)
            values["exposure_time"] = exif.get("ExposureTime")

            iso = exif.get("ISOSpeedRatings")
            if iso:
                values["iso"] = int(iso)

            gps_coords = _get_gps_coord(exif)
            if gps_coords:
                values["gps"] = gps_coords
                lat, lon = values["gps"]
                location_info = _get_location(lat, lon)
                neighbourhood = location_info.get("neighbourhood")
                city = location_info.get("city")
                country = location_info.get("country")
                if city and country:
                    location_parts = [
                        part for part in [neighbourhood, city, country] if part
                    ]
                    values["location"] = ", ".join(location_parts)

        except (UnidentifiedImageError, TypeError, FileNotFoundError) as e:
            warn_msg = f"Cannot extract EXIF from {loc}: {e}"
            logger.warning(warn_msg)
        except Exception as e:
            err_msg = f"Error processing EXIF for {loc}: {e}"
            logger.error(err_msg)
        return values

    def update_caption(self, new_caption: str, validated: bool = True) -> None:
        """Update caption and timestamps"""
        if validated:
            self.validated_caption = new_caption
        else:
            self.generated_caption = new_caption
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to flat dict for DataFrame insertion"""
        return {
            "image_id": self.image_id,
            "image_location": str(self.image_location),
            "filename": self.filename,
            "generated_caption": self.generated_caption,
            "validated_caption": self.validated_caption,
            "extracted_text": self.extracted_text,
            "make": self.make,
            "model": self.model,
            "software": self.software,
            "image_width": self.image_width,
            "image_length": self.image_length,
            "aperture_value": self.aperture_value,
            "focal_length": self.focal_length,
            "exposure_time": self.exposure_time,
            "brightness": self.brightness,
            "iso": self.iso,
            "gps": self.gps,
            "location": self.location,
            "processed": self.processed,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    model_config = {
        "json_encoders": {Path: str, datetime: lambda v: v.isoformat()},
        "extra": "forbid",
    }

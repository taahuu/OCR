"""Safe image loading and normalisation with Pillow."""

from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import Settings
from app.core.exceptions import CorruptDocumentError, ImageTooLargeError
from app.core.logging import get_logger

logger = get_logger("image")


class ImageService:
    """Opens untrusted image bytes and returns an OCR-ready RGB/L image."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load(self, data: bytes) -> Image.Image:
        """Decode ``data`` into a fully loaded, orientation-corrected image.

        Dimensions are checked from the header *before* pixel data is decoded
        so decompression bombs are rejected cheaply.

        Raises:
            CorruptDocumentError: the bytes are not a decodable image.
            ImageTooLargeError: the image exceeds the configured limits.
        """
        try:
            with warnings.catch_warnings():
                # Pillow warns (then errors at 2x) on suspiciously large images.
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image = Image.open(io.BytesIO(data))
                self._check_dimensions(image.size)
                image.load()
        except ImageTooLargeError:
            raise
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            if isinstance(exc, (Image.DecompressionBombError, Image.DecompressionBombWarning)):
                raise ImageTooLargeError() from exc
            raise CorruptDocumentError("The image file is corrupted or unreadable.") from exc
        except Exception as exc:  # noqa: BLE001 - Pillow raises many decode errors
            logger.info("Image decode failed: %s", type(exc).__name__)
            raise CorruptDocumentError("The image file is corrupted or unreadable.") from exc

        return self._normalise(image)

    def _check_dimensions(self, size: tuple[int, int]) -> None:
        width, height = size
        limit = self._settings.max_image_dimension
        if width <= 0 or height <= 0:
            raise CorruptDocumentError("The image has invalid dimensions.")
        if width > limit or height > limit or width * height > self._settings.max_image_pixels:
            raise ImageTooLargeError()

    @staticmethod
    def _normalise(image: Image.Image) -> Image.Image:
        """Apply EXIF orientation and convert to a plain 8-bit mode."""
        image = ImageOps.exif_transpose(image) or image

        mode = image.mode
        if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in image.info):
            # Flatten transparency onto white; black backgrounds ruin OCR.
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        if mode in ("I;16", "I;16L", "I;16B", "I"):
            # 16/32-bit grayscale: scale into 8 bits instead of clipping.
            values = np.asarray(image, dtype=np.float64)
            full_scale = 65535.0 if values.max() > 255 else 255.0
            scaled = np.rint(np.clip(values * (255.0 / full_scale), 0, 255)).astype(np.uint8)
            return Image.fromarray(scaled, mode="L")
        if mode in ("L", "RGB"):
            return image
        # 1, P, CMYK, YCbCr, LAB, HSV, F ...
        return image.convert("RGB")

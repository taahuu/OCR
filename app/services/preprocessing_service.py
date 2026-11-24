"""Conditional image preprocessing for OCR.

Tesseract already performs its own binarisation and works best on clean,
high-contrast pages, so aggressive preprocessing can *reduce* accuracy. This
service therefore:

* never modifies the input image (all operations return new images);
* applies each operation only when a cheap measurement says it is needed;
* is intended to be used as a *second attempt* after a low-confidence plain
  OCR pass (see ``DocumentService``), keeping whichever result is better.

Each ``enhance`` call reports the steps it applied so they can be returned in
the API response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from app.core.config import Settings


@dataclass(frozen=True)
class PreprocessResult:
    """A processed image and the names of the steps that were applied."""

    image: Image.Image
    steps: list[str] = field(default_factory=list)


class PreprocessingService:
    """Measures image quality and applies only the corrections that are needed."""

    # Algorithm tuning parameters (not deployment configuration).
    MAX_UPSCALE = 3.0
    LOW_CONTRAST_DELTA = 100  # (paper level - ink level) below this => stretch contrast
    NOISE_SIGMA_THRESHOLD = 4.0  # estimated gray-level noise std => denoise
    IMPULSE_RATIO_THRESHOLD = 0.004  # share of salt-and-pepper pixels => denoise
    UNEVEN_BACKGROUND_STD = 25.0  # std of block backgrounds => binarise adaptively
    MIN_SKEW_DEGREES = 0.3
    MAX_SKEW_DEGREES = 10.0
    SKEW_IMPROVEMENT_RATIO = 1.05
    BLANK_INK_RATIO = 0.0001

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------ API

    @staticmethod
    def to_grayscale(image: Image.Image) -> Image.Image:
        """Return a grayscale copy (or the same object if already grayscale)."""
        return image if image.mode == "L" else image.convert("L")

    def is_blank(self, image: Image.Image) -> bool:
        """True when the page has (almost) no dark pixels, so OCR is pointless."""
        gray = np.asarray(self.to_grayscale(image))
        background = float(np.median(gray))
        ink_ratio = float(np.mean(gray < background * 0.6))
        return ink_ratio < self.BLANK_INK_RATIO

    def enhance(self, image: Image.Image, rotation_degrees: int = 0) -> PreprocessResult:
        """Return an OCR-optimised copy of ``image`` plus the steps applied.

        Args:
            image: Source image (left unmodified).
            rotation_degrees: Clockwise rotation (90/180/270) to apply first,
                typically supplied by Tesseract orientation detection.
        """
        steps: list[str] = []
        gray = np.asarray(self.to_grayscale(image)).copy()

        if rotation_degrees in (90, 180, 270):
            # np.rot90 is counter-clockwise, so rotate by the complementary count.
            gray = np.ascontiguousarray(np.rot90(gray, k=(360 - rotation_degrees) // 90))
            steps.append(f"rotate_{rotation_degrees}")

        # Denoise first, at native resolution (upscaling would turn speckle into blobs).
        if (
            self._estimate_noise(gray) > self.NOISE_SIGMA_THRESHOLD
            or self._impulse_ratio(gray) > self.IMPULSE_RATIO_THRESHOLD
        ):
            gray = cv2.medianBlur(gray, 3)
            steps.append("denoise")

        if self._is_low_contrast(gray):
            gray = self._stretch_contrast(gray)
            steps.append("contrast")

        gray, scaled = self._upscale_if_small(gray)
        if scaled:
            steps.append("upscale")

        angle = self._estimate_skew(gray)
        if angle is not None:
            gray = self._rotate_small(gray, angle)
            steps.append(f"deskew_{angle:+.1f}deg")

        if self._has_uneven_background(gray):
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )
            steps.append("binarize")

        return PreprocessResult(image=Image.fromarray(gray), steps=steps)

    # ------------------------------------------------------------ measurements

    def _upscale_if_small(self, gray: np.ndarray) -> tuple[np.ndarray, bool]:
        long_side = max(gray.shape)
        target = self._settings.preprocess_min_long_side
        if long_side >= target:
            return gray, False
        factor = min(self.MAX_UPSCALE, target / long_side)
        if factor < 1.05:
            return gray, False
        resized = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        return resized, True

    def _is_low_contrast(self, gray: np.ndarray) -> bool:
        """Compare the paper level (median) with the ink level (0.5th percentile).

        Plain percentile ranges are unsuitable here: on a mostly-white page
        with sparse text, the 2nd percentile is already white.
        """
        ink, paper = np.percentile(gray, 0.5), np.median(gray)
        return (paper - ink) < self.LOW_CONTRAST_DELTA

    @staticmethod
    def _stretch_contrast(gray: np.ndarray) -> np.ndarray:
        ink, paper = np.percentile(gray, 0.5), np.median(gray)
        if paper - ink < 1:
            return gray
        stretched = (gray.astype(np.float32) - ink) * (255.0 / (paper - ink))
        return np.clip(stretched, 0, 255).astype(np.uint8)

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        """Robust noise estimate (Immerkaer's filter with a median/MAD scale).

        Using the median keeps the sparse text edges from inflating the result.
        """
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
        response = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        mad = float(np.median(np.abs(response)))
        return mad / 0.6745 / 6.0  # kernel response std is 6 * sigma

    @staticmethod
    def _impulse_ratio(gray: np.ndarray) -> float:
        """Share of isolated pixels that differ strongly from their 3x3 median.

        Catches salt-and-pepper speckle, which the Gaussian estimate above
        cannot see because most pixels are untouched.
        """
        difference = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
        return float(np.mean(difference > 80))

    @staticmethod
    def _has_uneven_background(gray: np.ndarray, grid: int = 8) -> bool:
        """Detect shadows / gradients by comparing local background levels."""
        scale = 512 / max(gray.shape)
        small = cv2.resize(gray, None, fx=min(scale, 1.0), fy=min(scale, 1.0),
                           interpolation=cv2.INTER_AREA)
        height, width = small.shape
        backgrounds = []
        for row in range(grid):
            for col in range(grid):
                block = small[
                    row * height // grid : (row + 1) * height // grid,
                    col * width // grid : (col + 1) * width // grid,
                ]
                if block.size:
                    backgrounds.append(np.percentile(block, 90))
        return float(np.std(backgrounds)) > PreprocessingService.UNEVEN_BACKGROUND_STD

    def _estimate_skew(self, gray: np.ndarray) -> float | None:
        """Estimate page skew (degrees) with a projection-profile search.

        Returns the rotation (degrees, counter-clockwise positive, as used by
        OpenCV) that straightens the text, or ``None`` if the page is not
        meaningfully skewed or the estimate is unreliable.
        """
        scale = min(1.0, 1000 / max(gray.shape))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Adaptive (local) thresholding so uneven lighting does not look like skew.
        binary = cv2.adaptiveThreshold(
            small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        if cv2.countNonZero(binary) < 200:
            return None

        def score(angle: float) -> float:
            height, width = binary.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            rotated = cv2.warpAffine(binary, matrix, (width, height), flags=cv2.INTER_NEAREST)
            return float(np.var(rotated.sum(axis=1, dtype=np.float64)))

        coarse = np.arange(-self.MAX_SKEW_DEGREES, self.MAX_SKEW_DEGREES + 0.01, 1.0)
        best = max(coarse, key=score)
        fine = np.arange(best - 1.0, best + 1.01, 0.1)
        best = max(fine, key=score)

        if abs(best) < self.MIN_SKEW_DEGREES:
            return None
        if score(best) < score(0.0) * self.SKEW_IMPROVEMENT_RATIO:
            return None
        return float(round(best, 1))

    @staticmethod
    def _rotate_small(gray: np.ndarray, angle: float) -> np.ndarray:
        height, width = gray.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(
            gray,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

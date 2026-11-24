import numpy as np
import pytest
from PIL import Image

from app.services.preprocessing_service import PreprocessingService
from tests import document_factory as docs
from tests.conftest import make_settings

LINES = docs.SAMPLE_LINES


@pytest.fixture
def service() -> PreprocessingService:
    return PreprocessingService(make_settings())


@pytest.fixture
def page() -> Image.Image:
    return docs.render_text_image(LINES).convert("L")


def _gradient(image: Image.Image, low: float) -> Image.Image:
    array = np.asarray(image).astype(float) * np.linspace(low, 1, image.width)[None, :]
    return Image.fromarray(array.astype("uint8"))


class TestBasics:
    def test_grayscale_conversion(self, service):
        rgb = Image.new("RGB", (10, 10), (200, 50, 50))
        assert service.to_grayscale(rgb).mode == "L"

    def test_grayscale_returns_same_object_when_already_gray(self, service, page):
        assert service.to_grayscale(page) is page

    def test_input_image_is_never_modified(self, service, page):
        before = np.asarray(page).copy()
        service.enhance(docs.render_text_image(LINES, rotate_degrees=4).convert("L"))
        service.enhance(page, rotation_degrees=90)
        assert np.array_equal(np.asarray(page), before)

    def test_blank_page_detected(self, service):
        assert service.is_blank(Image.new("L", (800, 1000), 255))

    def test_page_with_text_is_not_blank(self, service, page):
        assert not service.is_blank(page)


class TestConditionalSteps:
    def test_clean_page_needs_no_processing(self, service, page):
        assert service.enhance(page).steps == []

    def test_small_image_is_upscaled(self, service):
        small = docs.render_text_image(LINES).convert("L").resize((600, 775))
        result = service.enhance(small)
        assert "upscale" in result.steps
        assert max(result.image.size) >= 1500

    def test_upscale_is_capped(self, service):
        tiny = Image.new("L", (100, 100), 255)
        result = service.enhance(tiny)
        assert max(result.image.size) <= 100 * service.MAX_UPSCALE + 1

    def test_low_contrast_is_stretched(self, service, page):
        faded = Image.fromarray((120 + np.asarray(page) * 0.2).astype("uint8"))
        result = service.enhance(faded)
        assert "contrast" in result.steps
        assert np.asarray(result.image).min() < 30  # ink restored to dark

    def test_gaussian_noise_is_denoised(self, service, page):
        rng = np.random.default_rng(1)
        noisy = np.clip(np.asarray(page) + rng.normal(0, 25, (page.height, page.width)), 0, 255)
        assert "denoise" in service.enhance(Image.fromarray(noisy.astype("uint8"))).steps

    def test_salt_and_pepper_noise_is_denoised(self, service, page):
        rng = np.random.default_rng(2)
        array = np.asarray(page).copy()
        mask = rng.random(array.shape)
        array[mask < 0.02] = 0
        array[mask > 0.98] = 255
        assert "denoise" in service.enhance(Image.fromarray(array)).steps

    def test_uneven_lighting_is_binarised(self, service, page):
        result = service.enhance(_gradient(page, 0.25))
        assert "binarize" in result.steps
        assert set(np.unique(np.asarray(result.image))) <= {0, 255}

    def test_shadow_does_not_cause_false_deskew(self, service, page):
        steps = service.enhance(_gradient(page, 0.25)).steps
        assert not any(step.startswith("deskew") for step in steps)


class TestOrientation:
    @pytest.mark.parametrize("angle", [3.0, -4.0, 6.0])
    def test_skew_is_detected_and_corrected(self, service, angle):
        skewed = docs.render_text_image(LINES, rotate_degrees=angle).convert("L")
        result = service.enhance(skewed)
        deskew_steps = [s for s in result.steps if s.startswith("deskew")]
        assert deskew_steps, f"no deskew for {angle} degrees: {result.steps}"
        applied = float(deskew_steps[0].split("_")[1].removesuffix("deg"))
        assert applied == pytest.approx(-angle, abs=0.6)

    def test_straight_page_is_not_deskewed(self, service, page):
        assert not any(s.startswith("deskew") for s in service.enhance(page).steps)

    @pytest.mark.parametrize("rotation", [90, 180, 270])
    def test_explicit_rotation_restores_original(self, service, page, rotation):
        # PIL rotates counter-clockwise; undo with a clockwise `rotation` request.
        rotated = page.rotate(rotation, expand=True, fillcolor=255)
        restored = service.enhance(rotated, rotation_degrees=rotation).image
        assert restored.size == page.size
        difference = np.abs(np.asarray(restored).astype(int) - np.asarray(page).astype(int))
        assert difference.mean() < 1.0

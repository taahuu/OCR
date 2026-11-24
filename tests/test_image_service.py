import io

import pytest
from PIL import Image

from app.core.exceptions import CorruptDocumentError, ImageTooLargeError
from app.services.image_service import ImageService
from tests import document_factory as docs
from tests.conftest import make_settings


@pytest.fixture
def service() -> ImageService:
    return ImageService(make_settings())


def _bytes(image: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


class TestLoad:
    def test_loads_png_and_jpeg(self, service):
        source = docs.render_text_image(["hello"], size=(300, 200))
        assert service.load(_bytes(source, "PNG")).size == (300, 200)
        assert service.load(_bytes(source, "JPEG")).size == (300, 200)

    def test_transparent_png_is_flattened_onto_white(self, service):
        rgba = Image.new("RGBA", (40, 40), (0, 0, 0, 0))  # fully transparent
        result = service.load(_bytes(rgba))
        assert result.mode == "RGB"
        assert result.getpixel((5, 5)) == (255, 255, 255)

    def test_palette_image_with_transparency(self, service):
        palette = Image.new("P", (20, 20), 0)
        result = service.load(_bytes(palette, transparency=0))
        assert result.mode == "RGB"
        assert result.getpixel((1, 1)) == (255, 255, 255)

    @pytest.mark.parametrize("mode", ["1", "P", "CMYK", "LA"])
    def test_other_modes_become_rgb_or_gray(self, service, mode):
        image = Image.new(mode, (20, 20))
        fmt = "JPEG" if mode == "CMYK" else "PNG"
        result = service.load(_bytes(image, fmt))
        assert result.mode in ("RGB", "L")

    def test_sixteen_bit_grayscale_is_scaled_not_clipped(self, service):
        image = Image.new("I;16", (30, 30), 40000)
        result = service.load(_bytes(image))
        assert result.mode == "L"
        assert result.getpixel((0, 0)) == round(40000 / 65535 * 255)  # 156, not clipped to 255

    def test_exif_orientation_is_applied(self, service):
        image = Image.new("RGB", (200, 100), "white")
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 CW to display
        result = service.load(_bytes(image, "JPEG", exif=exif))
        assert result.size == (100, 200)

    def test_original_bytes_are_untouched(self, service):
        data = _bytes(docs.render_text_image(["x"], size=(60, 60)))
        before = bytes(data)
        service.load(data)
        assert data == before


class TestRejection:
    def test_corrupted_image(self, service):
        good = _bytes(docs.render_text_image(["text"], size=(300, 300)))
        with pytest.raises(CorruptDocumentError):
            service.load(good[: len(good) // 2])

    def test_garbage_with_png_signature(self, service):
        with pytest.raises(CorruptDocumentError):
            service.load(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    def test_not_an_image(self, service):
        with pytest.raises(CorruptDocumentError):
            service.load(b"definitely not an image")

    def test_pixel_limit(self):
        service = ImageService(make_settings(max_image_pixels=10_000))
        with pytest.raises(ImageTooLargeError):
            service.load(_bytes(Image.new("RGB", (200, 200))))

    def test_dimension_limit(self):
        service = ImageService(make_settings(max_image_dimension=100))
        with pytest.raises(ImageTooLargeError):
            service.load(_bytes(Image.new("RGB", (101, 10))))

    def test_decompression_bomb_is_rejected_from_the_header(self):
        # Highly compressible 20000x20000 image: tiny file, 400M pixels.
        bomb = _bytes(Image.new("1", (20000, 20000)), "PNG", optimize=False)
        assert len(bomb) < 5_000_000
        service = ImageService(make_settings())
        with pytest.raises(ImageTooLargeError):
            service.load(bomb)

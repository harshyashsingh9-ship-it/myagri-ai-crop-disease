from pathlib import Path
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def find_images(dataset_path):
    """Find all supported image files recursively."""
    dataset_path = Path(dataset_path)

    return [
        path for path in dataset_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_image(image_path):
    """Load an image as RGB."""
    return Image.open(image_path).convert("RGB")


def validate_image(image_path, min_size=100):
    """Validate that an image can be opened and has sufficient resolution."""
    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            width, height = image.size

        return width >= min_size and height >= min_size

    except Exception:
        return False

"""
Image preprocessing utilities for the crop disease prediction system.
"""

from PIL import Image


def load_image(image_path):
    """Load an image from the given path."""
    image = Image.open(image_path).convert("RGB")
    return image


def validate_image(image):
    """Basic image validation."""
    if image is None:
        raise ValueError("Invalid image.")

    width, height = image.size

    if width < 100 or height < 100:
        raise ValueError("Image resolution is too low.")

    return True

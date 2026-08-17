from PIL import Image
from src.data.preprocessing import validate_image


def test_valid_image():

    image_path = "tests/test_leaf.jpg"

    image = Image.new(
        "RGB",
        (224, 224)
    )

    image.save(image_path)

    assert validate_image(image_path)

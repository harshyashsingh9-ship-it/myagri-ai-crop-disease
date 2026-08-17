import torch
from PIL import Image

from src.data.augmentation import get_validation_transforms


class Predictor:

    def __init__(self, model, class_names, device=None):

        self.model = model
        self.class_names = class_names

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = get_validation_transforms()

    def predict(self, image):

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        return {
            "class": self.class_names[prediction.item()],
            "confidence": float(confidence.item())
        }

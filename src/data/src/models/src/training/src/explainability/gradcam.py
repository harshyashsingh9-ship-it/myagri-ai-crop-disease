"""
Grad-CAM explainability module.

This module will generate visual explanations showing
which regions of a leaf image contributed most strongly
to the model prediction.
"""


class GradCAM:

    def __init__(self, model):
        self.model = model

    def explain(self, image):
        """
        Generate Grad-CAM explanation.

        Implementation will be added after the base
        classifier has been trained and validated.
        """
        raise NotImplementedError(
            "Grad-CAM implementation is pending."
        )

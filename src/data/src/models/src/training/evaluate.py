import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


def evaluate_model(
    model,
    test_loader,
    class_names,
    device=None
):

    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            preds = outputs.argmax(
                dim=1
            ).cpu().numpy()

            predictions.extend(preds)
            actual.extend(labels.numpy())

    print(
        classification_report(
            actual,
            predictions,
            target_names=class_names,
            zero_division=0
        )
    )

    return confusion_matrix(
        actual,
        predictions
    )

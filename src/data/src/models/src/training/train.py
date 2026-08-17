import torch
from pathlib import Path

from src.models.efficientnet import create_model


def train_model(
    train_loader,
    validation_loader,
    num_classes,
    epochs=10,
    learning_rate=0.001
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = create_model(
        num_classes=num_classes,
        pretrained=True
    )

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    best_accuracy = 0.0

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        validation_accuracy = evaluate(
            model,
            validation_loader,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {running_loss:.4f} "
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            Path("models").mkdir(
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                "models/best_efficientnet_b0.pth"
            )

    return model


def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total if total else 0

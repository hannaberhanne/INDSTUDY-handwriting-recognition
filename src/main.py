import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms for EMNIST images
transform_train = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)
])

transform_test = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)
])

# Load EMNIST Letters dataset
train_data = datasets.EMNIST(root='.', split='letters', train=True, download=True, transform=transform_train)
test_data = datasets.EMNIST(root='.', split='letters', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Basic CNN model for letter classification
class LetterCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(5 * 5 * 64, 128)  # assumes input dims after pooling
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # (28 - 3 + 1) = 26
        x = F.max_pool2d(x, 2)          # 13x13
        x = F.relu(self.conv2(x))       # 11x11 -> pool -> ~5x5
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = LetterCNN(num_classes=27).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train the model or load existing one
model_path = 'emnist_cnn.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded saved model from {model_path}")
else:
    print("Training new model...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5 - Avg Loss: {total_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Evaluate on test set
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

evaluate_model()

# Run prediction on a custom image
def predict_custom_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Couldn't load image: {path}")
        return

    # Threshold + invert (so letter is white on black bg)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return

    # Use largest contour (likely the letter)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    margin = 8  # instead of 4
    x_start = max(x - margin, 0)
    y_start = max(y - margin, 0)
    x_end = min(x + w + margin, binary.shape[1])
    y_end = min(y + h + margin, binary.shape[0])
    cropped = binary[y_start:y_end, x_start:x_end]

    # Optional smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)

    # Resize and normalize
    resized = cv2.resize(cropped, (28, 28))
    normalized = resized.astype('float32') / 255.0
    inverted = 1 - normalized
    tensor = torch.tensor(inverted).unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        predicted = output.argmax(dim=1).item()
        predicted_letter = chr(predicted + 96)  # EMNIST: 1 = 'a'

    print(f"Predicted Letter: {predicted_letter}")
    plt.imshow(inverted, cmap='gray')
    plt.title(f"Prediction: {predicted_letter}")
    plt.axis('off')
    plt.savefig("prediction_result.png")
    print("Result image saved as prediction_result.png")

# Test with one of the letter images
predict_custom_image('../data/letters/K.jpg')
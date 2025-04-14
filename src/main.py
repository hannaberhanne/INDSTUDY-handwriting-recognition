import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for training and testing (EMNIST samples are centered and resized)
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)  # Invert colors: background 0, ink 1
])

transform_test = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)
])

# Load EMNIST Letters dataset (downloads if necessary)
train_data = datasets.EMNIST(root='.', split='letters', train=True, download=True, transform=transform_train)
test_data = datasets.EMNIST(root='.', split='letters', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Define the CNN architecture
class LetterCNN(nn.Module):
    def __init__(self, num_classes):
        super(LetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)   # Input: 28x28 -> Output: 26x26
        self.conv2 = nn.Conv2d(32, 64, 3)   # Output size changes after pooling
        # After 2 conv layers and 2 max-poolings, the feature map is roughly 5x5 with 64 channels (5*5*64 = 1600)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # roughly 13x13
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # roughly 5x5
        x = x.view(x.size(0), -1)  # flattened dimension auto-calculated
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = LetterCNN(num_classes=27).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model_path = 'emnist_cnn.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Loaded pre-trained model from", model_path)
else:
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), model_path)
    print("Model saved as", model_path)

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

evaluate()

# Predict a custom image with bounding-box centering and generate a result picture
def predict_image(img_path):
    # Load the custom image (should be grayscale, same folder as main.py)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image '{img_path}'. Check the file path and integrity.")
        return
    
    # Convert the image to binary using Otsu's thresholding (inversion included)
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours on the binary image
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found; check the image content.")
        return

    # Use the largest contour (assumed to be the letter) to compute the bounding box
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image to the bounding box
    cropped = bin_img[y:y+h, x:x+w]
    
    # Optional: apply morphological closing to connect any broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)
    
    # Resize the cropped image to 28x28 pixels (EMNIST standard)
    processed_img = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize the processed image to the range [0, 1]
    processed_img = processed_img.astype('float32') / 255.0

    # Convert to tensor with shape (1, 1, 28, 28)
    img_tensor = torch.tensor(processed_img).unsqueeze(0).unsqueeze(0).to(device)
    
    model.eval()
    output = model(img_tensor)
    pred = output.argmax(dim=1)
    predicted_letter = chr(pred.item() + 96)  # EMNIST: 1 maps to 'a'
    print("Predicted Letter:", predicted_letter)
    
    plt.imshow(processed_img, cmap='gray')
    plt.title(f"Predicted: {predicted_letter}")
    plt.axis('off')
    plt.savefig("prediction_result.png")
    print("Prediction result saved as prediction_result.png")

# Use 'H.jpg' (your custom image file in the same folder as main.py)
predict_image('H.jpg')
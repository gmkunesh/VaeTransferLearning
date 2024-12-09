import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 5

# Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, loader, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader)}")

# Train the model
train(model, train_loader, criterion, optimizer)
# Save the initial model
torch.save(model.state_dict(), "initial_mnist_cnn.pth")


# Filter dataset for digits 0, 1, and 2
class FilteredMNIST(datasets.MNIST):
    def __init__(self, *args, allowed_digits=None, **kwargs):
        kwargs['download'] = True  # Ensure download is enabled
        super().__init__(*args, **kwargs)
        if allowed_digits is not None:
            mask = [label in allowed_digits for label in self.targets]
            self.data = self.data[mask]
            self.targets = self.targets[mask]

allowed_digits = [0, 1, 2]
train_dataset_3digits = FilteredMNIST(root='./data', train=True, transform=transform, allowed_digits=allowed_digits)
train_loader_3digits = DataLoader(train_dataset_3digits, batch_size=batch_size, shuffle=True)
# Modify the final layer to classify 3 digits
model.fc2 = nn.Linear(32, 3)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train on the filtered dataset
train(model, train_loader_3digits, criterion, optimizer)
torch.save(model.state_dict(), "fine_tuned_3digits.pth")


# Map 7 irrelevant digits to a new class ("other")
class GroupedMNIST(datasets.MNIST):
    def __init__(self, *args, relevant_digits=None, **kwargs):
        kwargs['download'] = True  # Ensure download is enabled
        super().__init__(*args, **kwargs)
        self.relevant_digits = relevant_digits if relevant_digits is not None else []
        self.targets = self.targets.clone()
        for i in range(10):
            if i not in self.relevant_digits:
                self.targets[self.targets == i] = len(self.relevant_digits)

relevant_digits = [0, 1, 2]
train_dataset_grouped = GroupedMNIST(root='./data', train=True, transform=transform, relevant_digits=relevant_digits)
train_loader_grouped = DataLoader(train_dataset_grouped, batch_size=batch_size, shuffle=True)
# Modify the final layer to classify 4 classes (3 relevant + 1 "other")
model.fc2 = nn.Linear(32, 4)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train on the grouped dataset
weights = torch.tensor([1.0, 1.0, 1.0, 0.25])  # Higher weight for "Other"
criterion = nn.CrossEntropyLoss(weight=weights)
train(model, train_loader_grouped, criterion, optimizer)
torch.save(model.state_dict(), "fine_tuned_grouped.pth")


# Create a filtered test dataset for digits 0, 1, and 2
test_dataset_filtered = FilteredMNIST(root='./data', train=False, transform=transform, allowed_digits=[0, 1, 2])
test_loader_filtered = DataLoader(test_dataset_filtered, batch_size=batch_size, shuffle=False)

# Function to evaluate a model
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)

# Function to plot test results
def plot_test_results(labels, preds, model_name):
    correct = (labels == preds).sum().item()
    total = labels.shape[0]
    accuracy = correct / total * 100
    print(f"{model_name} Test Accuracy: {accuracy:.2f}%")
    
    # Optional: You could also create a bar chart of accuracy for visual representation
    plt.bar([model_name], [accuracy], color='blue')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy of Models')
    plt.show()

# Evaluate Model 1 (3-digit model)
model_3digits = SimpleCNN()
model_3digits.fc2 = nn.Linear(32, 3)
model_3digits.load_state_dict(torch.load("fine_tuned_3digits.pth"))
labels_3digits, preds_3digits = evaluate_model(model_3digits, test_loader_filtered)
plot_test_results(labels_3digits, preds_3digits, "3-Digit Model")

# Evaluate Model 2 (grouped model)
model_grouped = SimpleCNN()
model_grouped.fc2 = nn.Linear(32, 4)
model_grouped.load_state_dict(torch.load("fine_tuned_grouped.pth"))
labels_grouped, preds_grouped = evaluate_model(model_grouped, test_loader_filtered)
plot_test_results(labels_grouped, preds_grouped, "Grouped Model")


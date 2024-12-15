import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 25

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
            #print(images.dtype)
            #print(labels.dtype)
            images = images.to(torch.float32)
            labels = labels.to(torch.int64)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader)}")

# Train the model
print('Training Full Model')
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
print('Training 3 Target Model')
train(model, train_loader_3digits, criterion, optimizer)
torch.save(model.state_dict(), "fine_tuned_3digits.pth")

# Joe's Dataset loader
class JoesDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform = None):
        super().__init__()
        self.data = data.unsqueeze(1).to(torch.uint8)
        self.targets = targets.to(torch.int64)
        self.transform = transform  # Store the transform as an attribute
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the data and target at index `idx`
        image, label = self.data[idx], self.targets[idx]
        
        # Apply the transform if it's provided
        #if self.transform:
            #image = self.transform(image)
        
        return image, label

# Joe's section for loading in Odds for End2End Finetuning
odds_data = torch.load('odd_Dataset_train.pt')
odds_testing = torch.load('odd_Dataset_test.pt')
core_data = torch.load('MNISTFull_Coreset.pt')

# Relabel data for odd set
for i in range(len(odds_data)):
    if odds_data.targets[i] == 1:
        odds_data.targets[i] = 0
    elif odds_data.targets[i] == 3:
        odds_data.targets[i] = 1
    elif odds_data.targets[i] == 5:
        odds_data.targets[i] = 2
    elif odds_data.targets[i] == 7:
        odds_data.targets[i] = 3
    elif odds_data.targets[i] == 9:
        odds_data.targets[i] = 4

for i in range(len(odds_testing)):
    if odds_testing.targets[i] == 1:
        odds_testing.targets[i] = 0
    elif odds_testing.targets[i] == 3:
        odds_testing.targets[i] = 1
    elif odds_testing.targets[i] == 5:
        odds_testing.targets[i] = 2
    elif odds_testing.targets[i] == 7:
        odds_testing.targets[i] = 3
    elif odds_testing.targets[i] == 9:
        odds_testing.targets[i] = 4

    # if odds_dataset.targets[i] % 2 == 1:
    #     odds_dataset.targets[i] = np.floor(odds_dataset.targets[i]/2)

# Relabel data for coreset set
for i in range(len(core_data)):
    if core_data.targets[i] == 1:
        core_data.targets[i] = 0
    elif core_data.targets[i] == 3:
        core_data.targets[i] = 1
    elif core_data.targets[i] == 5:
        core_data.targets[i] = 2
    elif core_data.targets[i] == 7:
        core_data.targets[i] = 3
    elif core_data.targets[i] == 9:
        core_data.targets[i] = 4
    else:
        core_data.targets[i] = 5

# Construct Datasets
Joetransform = transforms.Compose(transforms.Normalize((0.5,), (0.5,)))
odds_dataset = JoesDataset(data = odds_data.tensors[0], targets=odds_data.targets, transform=Joetransform)
core_dataset = JoesDataset(data = core_data.tensors[0], targets=core_data.targets, transform=Joetransform)

odds_testset = JoesDataset(data = odds_testing.tensors[0], targets=odds_testing.targets, transform=Joetransform)
odds_testloader = DataLoader(odds_testset, batch_size=batch_size, shuffle=True)

# Do Odds Only Model
odds_dataloader = DataLoader(odds_dataset, batch_size=batch_size, shuffle=True)

model.fc2 = nn.Linear(32,5)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train on the filtered dataset
print('Training Joes Odd Model')
train(model, odds_dataloader, criterion, optimizer)
torch.save(model.state_dict(), "JoeOdds.pth")

# Joe's Odds + Coreset Constant Schedule End2End
constant_dataset = JoesDataset(data = torch.cat((odds_data.tensors[0], core_data.tensors[0]),0),
                               targets = torch.cat((odds_data.targets, core_data.targets),0),
                               transform = Joetransform)
constant_dataloader = DataLoader(constant_dataset, batch_size=batch_size, shuffle=True)

model.fc2 = nn.Linear(32,6)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train on the filtered dataset
print('Training Constant Mixed Model')
train(model, constant_dataloader, criterion, optimizer)
torch.save(model.state_dict(), "JoeConstant.pth")

# Training loop
def Joetrain(model, oddset, coreset, method, criterion, optimizer):
    model.train()

    mix_ratio = torch.linspace(0, len(coreset)-1, epochs, dtype = torch.int)

    if method == 'Front':
        mix_ratio = torch.flip(mix_ratio,[0])

    for epoch in range(epochs):
        running_loss = 0.0

        # Mix together datasets
        mix_indices = torch.randint(0, len(coreset), [mix_ratio[epoch].tolist()])
        
        epoch_data = torch.cat([coreset.tensors[0][mix_indices,:,:],oddset.tensors[0]], 0)
        epoch_targets = torch.cat([coreset.targets[mix_indices],oddset.targets],0)

        epoch_dataset = JoesDataset(data = epoch_data, targets = epoch_targets, transform = Joetransform)
        loader = DataLoader(epoch_dataset, batch_size = batch_size, shuffle=True)
        
        for images, labels in loader:
            #print(images.dtype)
            #print(labels.dtype)
            images = images.to(torch.float32)
            labels = labels.to(torch.int64)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader)}")

model.fc2 = nn.Linear(32,6)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train on the filtered dataset
print('Training Joes Backloaded Model')
Joetrain(model, odds_data, core_data, 'Back', criterion, optimizer)
torch.save(model.state_dict(), "JoeOddsBack.pth")

model.fc2 = nn.Linear(32,6)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train on the filtered dataset
print('Training Joes Frontloaded Model')
Joetrain(model, odds_data, core_data, 'Front', criterion, optimizer)
torch.save(model.state_dict(), "JoeOddsFront.pth")

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
print('Training 3 Digit with Grouping Model')
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

# # Function to plot test results
# def plot_test_results(labels, preds, model_name):
#     correct = (labels == preds).sum().item()
#     total = labels.shape[0]
#     accuracy = correct / total * 100
#     print(f"{model_name} Test Accuracy: {accuracy:.2f}%")
    
#     # Optional: You could also create a bar chart of accuracy for visual representation
#     plt.bar([model_name], [accuracy], color='blue')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Test Accuracy of Models')
#     plt.show()

# # Evaluate Model 1 (3-digit model)
# model_3digits = SimpleCNN()
# model_3digits.fc2 = nn.Linear(32, 3)
# model_3digits.load_state_dict(torch.load("fine_tuned_3digits.pth"))
# labels_3digits, preds_3digits = evaluate_model(model_3digits, test_loader_filtered)
# plot_test_results(labels_3digits, preds_3digits, "3-Digit Model")

# # Evaluate Model 2 (grouped model)
# model_grouped = SimpleCNN()
# model_grouped.fc2 = nn.Linear(32, 4)
# model_grouped.load_state_dict(torch.load("fine_tuned_grouped.pth"))
# labels_grouped, preds_grouped = evaluate_model(model_grouped, test_loader_filtered)
# plot_test_results(labels_grouped, preds_grouped, "Grouped Model")

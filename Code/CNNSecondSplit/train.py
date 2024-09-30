import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sampleCNN import SimpleCNN
from dataset import CustomAudioDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Assuming you have already defined SimpleCNN, train_dataloader, val_dataloader, and test_dataloader

# Train the model with default parameters
def train_model(model, train_dataloader, criterion, optimizer, num_epochs = 50, batch_size = 50, device = device):
    # Update the batch size in DataLoader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss}')

    return model

# Define a function for model validation
def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    model.to(device)  # Ensure the model is on the correct device
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            # Transfer inputs and labels to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    val_accuracy = correct / total 
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Define transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])


dataset_path = r'C:\7. dönem\ml\project\split\equal_distribution_data'
# Create custom datasets for train, test, and validation
train_dataset = CustomAudioDataset(root_dir=os.path.join(dataset_path, 'train'), transform=transform)
test_dataset = CustomAudioDataset(root_dir=os.path.join(dataset_path, 'test'), transform=transform)
val_dataset = CustomAudioDataset(root_dir=os.path.join(dataset_path, 'validation'), transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the CNN model
model_default = SimpleCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_default.parameters(), lr=0.0001)

# Train the model with default parameters
num_epochs_default = 40
#train_model(model_default, train_dataloader, criterion, optimizer, num_epochs=num_epochs_default)

# Evaluate the model on the test set with default parameters
model_default.eval()
correct_default = 0
total_default = 0

with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_default(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_default += labels.size(0)
        correct_default += (predicted == labels).sum().item()

accuracy_default = correct_default / total_default
print(f'Validation Accuracy with Default Parameters: {accuracy_default * 100:.5f}%')



# Define hyperparameter grid
learning_rates = [0.001, 0.0001, 0.00001]
dropout_rates = [0.2, 0.5, 0.8]
batch_sizes = [16, 32, 50]
num_epochs_list = [10]

total_combinations = len(learning_rates) * len(dropout_rates) * len(num_epochs_list)
current_combination = 0



# Iterate over hyperparameter grid
for lr in learning_rates:
    for dropout_rate in dropout_rates:
        for batch_size in batch_sizes:
            for num_epochs in num_epochs_list:
                # Instantiate the CNN model
                model = SimpleCNN(dropout_rate=dropout_rate)
                model.to(device)

                # Define loss function and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # Train the model
                print("\n")
                print(f'Hyperparameter Combination - LR: {lr}, Dropout: {dropout_rate}, Batch Size: {batch_size}, Epochs: {num_epochs}')
                
                # Adjust the batch size in DataLoader
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                train_model(model, train_dataloader, criterion, optimizer, num_epochs=num_epochs, batch_size=batch_size)
                model.eval()
                # Validate the model on the validation set
                val_accuracy_current = validate_model(model, val_dataloader, criterion, device = device)

dropout_rate = 0.5
num_epochs = 10
lr = 0.001
batch_size = 32
 # Instantiate the CNN model
model = SimpleCNN(dropout_rate=dropout_rate).to(device)
model.dropout = nn.Dropout(dropout_rate)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
print("\n")
print(f'Hyperparameter Combination:')
print(f'  Learning Rate (LR): {lr}')
print(f'  Dropout Rate: {dropout_rate}')
print(f'  Number of Epochs: {num_epochs}')
print(f'  Batch size: {batch_size}')

model = train_model(model, train_dataloader, criterion, optimizer, num_epochs=num_epochs, device = device, batch_size = batch_size)

# Validate the model on the validation set


# Evaluate the model on the test set

correct = 0
total = 0

true_labels = []
predictions = []
model.eval()
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        true_labels.extend(labels.cpu().numpy())
        predictions.extend(predicted.cpu().numpy())

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.5f}%')


cm = confusion_matrix(true_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()
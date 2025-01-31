import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms ## torchvision.datasets contient des datasets prédifinis, 
#torchvisions.transforms permet d'appliquer des transformations aux images
from torch.utils.data import DataLoader
# Permet de créer des itérables pour charger les données par lots (batches).




class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # Input layer to hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Hyperparameters
input_size = 32 * 32 * 3  
hidden_size = 128  # Number of neurons in the hidden layer
output_size = 10  # 10 classes (digits 0-9)
learning_rate = 0.01
batch_size = 64
num_epochs = 5


# Charger le dataset CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(), # Normalise les valeurs entre 0 et 1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalise les valeurs des pixels en soustrayant la moyenne (mean) et en divisant par l'écart-type (std).
])
# Ici, (0.5, 0.5, 0.5) est utilisé pour les trois canaux de couleur (R, G, B), ce qui ramène les valeurs des pixels dans l'intervalle [-1, 1].
# Normalise les valeurs des pixels en soustrayant la moyenne et en divisant ar l'écart-type

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Initialize model, loss function, and optimizer
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')




# Testing the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
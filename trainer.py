import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Define CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Custom Dataset for Loading Corrected Images
class CustomCorrectionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        for filename in os.listdir(root):
            if filename.endswith(".png"):
                label = int(filename.split("_")[0])  # Extract label from filename
                self.images.append(os.path.join(root, filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("L")
        image = image.resize((28, 28))  # ✅ Ensure all images are 28x28
        
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

# Training Function
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Adjust epochs based on training or retraining
    is_retraining = os.path.exists("mnist_cnn.pt")
    if is_retraining:
        print("Model found. Retraining with fewer epochs.")
        epochs = 2  # Retraining
    else:
        print("No existing model found. Training from scratch.")
        epochs = 15  # Full training

    batch_size = 64
    lr = 1.0
    gamma = 0.7

    # Transformations (Normalization + Augmentations)
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Small rotations to generalize
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST Dataset
    mnist_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

    # Load Corrected Images if Available
    correction_dataset = None
    if os.path.exists("corrections/") and len(os.listdir("corrections/")) > 0:
        correction_dataset = CustomCorrectionDataset(root="corrections/", transform=transform)

    # Combine MNIST + Corrections
    if correction_dataset:
        train_dataset = ConcatDataset([mnist_dataset, correction_dataset])
        print(f"Training on MNIST + {len(correction_dataset)} corrected images.")
    else:
        train_dataset = mnist_dataset
        print("No corrections found. Training on MNIST only.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    model = Net().to(device)
    if is_retraining:
        model.load_state_dict(torch.load("mnist_cnn.pt"))  # Load pre-trained model for retraining

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        scheduler.step()

    # Save Updated Model
    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("Training complete. Model updated!")

    # Calculate Accuracy (only for full training)
    if not is_retraining:
        model.eval()
        test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transform), batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100. * correct / total
        print(f"Model Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train()
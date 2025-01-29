import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import models, datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os

def define_transforms(height, width):
    """Define transformations for training and testing datasets."""
    data_transforms = {
        'train': Compose([
            Resize((height, width)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': Compose([
            Resize((height, width)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    return data_transforms

def read_images(data_transforms):
    """Load training, validation, and test datasets."""
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'resumido')
    train_path = os.path.join(base_dir, 'train')
    validation_path = os.path.join(base_dir, 'validation')
    test_path = os.path.join(base_dir, 'test')

    # Verify directories exist
    for path in [train_path, validation_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")

    # Load datasets
    train_data = datasets.ImageFolder(train_path, transform=data_transforms['train'])
    validation_data = datasets.ImageFolder(validation_path, transform=data_transforms['test'])
    test_data = datasets.ImageFolder(test_path, transform=data_transforms['test'])

    return train_data, validation_data, test_data

class CNN:
    def __init__(self, train_data, validation_data, test_data, batch_size, save_dir="models"):
        """Initialize the CNN class with data loaders and device."""
        self.train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.validation_loader = data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
        self.test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def create_and_train_cnn(self, model_name, num_epochs, learning_rate, weight_decay, replications):
        """Train the model for multiple replications and return average accuracy."""
        soma = 0
        acc_max = 0
        for i in range(replications):
            model = self.create_model(model_name)
            optimizer = self.create_optimizer(model, learning_rate, weight_decay)
            criterion = self.create_criterion()
            self.train_model(model, self.train_loader, optimizer, criterion,
                            model_name, num_epochs, learning_rate, weight_decay, i)
            acc = self.evaluate_model(model, self.validation_loader)
            soma += acc
            if acc > acc_max:
                acc_max = acc
                iter_acc_max = i
        return soma / replications, iter_acc_max

    def create_model(self, model_name):
        """Create a pre-trained model with a modified final layer."""
        model_dict = {
            'alexnet': models.alexnet,
            'mobilenet_v3_large': models.mobilenet_v3_large,
            'mobilenet_v3_small': models.mobilenet_v3_small,
            'resnet18': models.resnet18,
            'resnet101': models.resnet101,
            'vgg11': models.vgg11,
            'vgg19': models.vgg19,
        }
        if model_name not in model_dict:
            raise ValueError(f"Model {model_name} not supported.")

        model = model_dict[model_name](weights='DEFAULT')
        for param in model.parameters():
            param.requires_grad = False

        # Modify the final layer based on the model architecture
        if model_name in ['alexnet', 'vgg11', 'vgg19']:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        elif model_name in ['mobilenet_v3_large', 'mobilenet_v3_small']:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        elif model_name in ['resnet18', 'resnet101']:
            model.fc = nn.Linear(model.fc.in_features, 2)

        return model.to(self.device)

    def create_optimizer(self, model, learning_rate, weight_decay):
        """Create an optimizer for the model."""
        params_to_update = [param for param in model.parameters() if param.requires_grad]
        return optim.SGD(params_to_update, lr=learning_rate, weight_decay=weight_decay)

    def create_criterion(self):
        """Create a loss function."""
        return nn.CrossEntropyLoss()

    def train_model(self, model, train_loader, optimizer, criterion, model_name, num_epochs, learning_rate, weight_decay, replication):
        """Train the model and save the best weights."""
        model.to(self.device)
        min_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            if train_loss < min_loss:
                min_loss = train_loss
                model_path = os.path.join(self.save_dir, f"{model_name}_{num_epochs}_{learning_rate}_{weight_decay}_{replication}.pth")
                torch.save(model.state_dict(), model_path)

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train the model for one epoch."""
        model.train()
        losses = []
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        return np.mean(losses)

    def evaluate_model(self, model, loader):
        """Evaluate the model on a dataset."""
        total = 0
        correct = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            output = model(X)
            _, y_pred = torch.max(output, 1)
            total += len(y)
            correct += (y_pred == y).sum().cpu().data.numpy()
        return correct / total
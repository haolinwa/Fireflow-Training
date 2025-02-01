import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
import copy
import numpy as np
import random

# Setup 

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

data_dir = 'dataset'  

# data augumentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Additional augmentation
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight translation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

# Calculate dataset sizes
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
valid_size = total_size - train_size

train_dataset, valid_dataset = random_split(
    full_dataset,
    [train_size, valid_size],
    generator=torch.Generator().manual_seed(42)
)

valid_dataset.dataset.transform = valid_transforms

batch_size = 16  

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define the custom classifier
class FireClassifier(nn.Module):
    def __init__(self, feature_extractor, num_classes=2):
        super(FireClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)  # ResNet18's final feature dimension is 512
        )
    
    def forward(self, x):
        with torch.no_grad():  # Freeze feature extractor
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)  
        out = self.classifier(features)
        return out


resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = nn.Sequential(*list(resnet18.children())[:-1]) 

#Initialize the custom model
model = FireClassifier(feature_extractor).to(device)

print(model)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001, weight_decay=1e-5)  #regularization

# Training par
num_epochs = 50
best_acc = 0.0
save_path = 'model.pth'
patience = 10 
counter = 0


def train_model():
    global best_acc, counter
    for epoch in range(num_epochs):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_running_loss / valid_size
        val_acc = val_running_corrects.double() / valid_size
        
        end_time = time.time()
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
              f'Time: {end_time - start_time:.0f}s')
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f'--> Best model saved with accuracy: {best_acc:.4f}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping triggered!')
                break
    
    print('Training complete.')
    print(f'Best validation accuracy: {best_acc:.4f}')
    print(f'Model saved to {save_path}')

if __name__ == '__main__':
    train_model()
   
    def load_model(model_path):
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
        
        model_loaded = FireClassifier(feature_extractor)
        model_loaded.load_state_dict(torch.load(model_path, map_location=device))
        model_loaded = model_loaded.to(device)
        model_loaded.eval()
        return model_loaded
    
    def evaluate_model(model_eval, data_loader_eval):
        model_eval.eval()
        running_corrects_eval = 0
        total_eval = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader_eval:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model_eval(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects_eval += torch.sum(preds == labels.data)
                total_eval += labels.size(0)
        
        accuracy_eval = running_corrects_eval.double() / total_eval
        print(f'Validation Accuracy: {accuracy_eval:.4f}')
    
    best_model = load_model(save_path)
    evaluate_model(best_model, valid_loader)

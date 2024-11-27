import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomRotation, RandomHorizontalFlip
import argparse

# Define Transformations for training and testing
train_transform = Compose([
    Resize((28, 28)),               # resize image to 28x28 pixels
    RandomRotation(10),             # Randomly rotate the image by up to 10 degrees
    RandomHorizontalFlip(),         # Randomly flip the image horizontally
    ToTensor(),                     # Convert the image to a tensor
    Normalize((0.5,), (0.5,))       # Normalize the tensor values to mean 0.5 and std 0.5
])

test_transform = Compose([
    Resize((28, 28)),               # resize image to 28x28 pixels
    ToTensor(),                     # Convert the image to a tensor
    Normalize((0.5,), (0.5,))       # Normalize the tensor values to mean 0.5 and std 0.5
])

# define image classification model
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),        # Convolutional layer with 32 filters
            nn.ReLU(),                      
            nn.Conv2d(32, 64, (3,3)),       # Convolutional layer with 64 filters
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),       # Convolutional layer with 64 filters
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),   # Adaptive average pooling to handle variable input sizes
            nn.Flatten(),                   # Flatten the output for the fully connected layer
            nn.Linear(64, 10),              # Fully connected layer with 10 output classes
        )
    
    def forward(self, x):
        return self.model(x) # Forward pass


def train_epoch(model, dataloader, loss_fn, optimiser, device):
    model.train()
    total_loss, correct = 0, 0
    for X, y in dataloader:                     # Iterate through batches
        X, y = X.to(device), y.to(device)       # Move data to the GPU/CPU
        optimiser.zero_grad()                   # Clear gradients
        yhat = model(X)                         # Forward pass
        loss = loss_fn(yhat, y)                 # Compute loss
        loss.backward()                         # Backpropagation
        optimiser.step()                        # Update model weights
        total_loss += loss.item()               # Accumulate loss

        correct += (torch.argmax(yhat, dim=1) == y).sum().item() # Count correct predictions
    return total_loss / len(dataloader), correct / len(dataloader.dataset) # Return loss and accuracy

def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():                       # Disable gradient computation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            yhat = model(X)                     # Forward pass
            loss = loss_fn(yhat, y)             # Compute loss
            total_loss += loss.item()           # Accumulate loss
            correct += (torch.argmax(yhat, dim=1) == y).sum().item() # Count correct predictions
    return total_loss / len(dataloader), correct / len(dataloader.dataset) # Return loss and accuracy

def load_model(path, model, device):
    try:
        with open(path, 'rb') as f:
            model.load_state_dict(load(f, weights_only=True)) # Load model weights
        model.to(device) # Move model to device
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

def save_model(path, model):
    with open(path, 'wb') as f:
        save(model.state_dict(), f) # Save model weights
    print(f"Model saved to {path}.")

def predict_image(model, image_path, transform, device):
    img = Image.open(image_path).convert("L")               # Open the image and convert to grayscale
    img_tensor = transform(img).unsqueeze(0).to(device)     # Transform and add batch dimension
    model.eval()
    with torch.no_grad():                                   # Disable gradient computation
        pred = torch.argmax(model(img_tensor), dim=1)       # Get the predicted class
    return pred.item()                                      # Return the predicted class


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available

    # initialise model, optimiser and loss function
    clf = ImageClassifier().to(device)
    opt = Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # load MNIST dataset with transformations
    train_data = datasets.MNIST(root="data", download=True, train=True, transform=train_transform)
    test_data = datasets.MNIST(root="data", download=True, train=False, transform=train_transform)

    # Split training data into train and validation datasets
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

    # Create data loaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    if args.train: # training mode
        best_val_loss = float('inf') # Track the best validation loss
        for epoch in range(10): # train for 10 epoch
            train_loss, train_acc = train_epoch(clf, train_loader, loss_fn, opt, device)
            val_loss, val_acc = validate_epoch(clf, val_loader, loss_fn, device)

            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model('best_model.pt', clf)
    
    if args.infer: # Inference mode
        load_model('best_model.pt', clf, device) # Load the saved model
        pred = predict_image(clf, args.infer, test_transform, device) # Predict the class of the input image
        print(f"Predicted digit for {args.infer}: {pred}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model') # Command-line option to train
    parser.add_argument('--infer', type=str, help="Path to the image for inference") # Command-line option to infer
    args = parser.parse_args()
    main(args)
    

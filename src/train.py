import numpy as np
import tqdm
import torch
import argparse
from data_processing import get_dataloader, create_tensor, create_dataframe
from model import create_model

def calculate_class_weights(true, nb_classes):
    """
    Calculate class weights based on the number of samples in each class.
    This is useful for handling class imbalance in datasets.
    Parameters
    ----------
    true : torch.Tensor
        Tensor of true labels of shape (n_samples,)
    nb_classes : int
        Number of classes in the dataset.

    Returns
    -------
    weights : torch.Tensor
        Tensor of class weights of shape (nb_classes,).
    """
    N = len(true)
    G = [np.count_nonzero(true == i) for i in range(nb_classes)]
    weights = [N / (nb_classes * G_i) if G_i != 0 else 0 for G_i in G]
    return weights

def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_L = []  # Train Losses
    val_L = [] # Validation losses
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for inputs, targets in train_loader_tqdm:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix(train_loss=running_loss / len(train_loader.dataset))

        # Evaluation on Validation Set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc='Validating', unit='batch')
            for inputs, targets in val_loader_tqdm:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            val_loader_tqdm.set_postfix(val_loss=val_loss / len(val_loader.dataset), val_acc=(correct / total) * 100)

        train_L.append(running_loss/len(train_loader.dataset))
        val_L.append(val_loss/len(val_loader.dataset))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader.dataset):.4f}, Val Loss: {val_loss/len(val_loader.dataset):.4f}, Val Acc: {(correct/total)*100:.2f}%")

    torch.save(model.state_dict(), 'model.pth')

    return model, train_L, val_L

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet on Painting Classification")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed image data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Load data
    df, image_names, _, painter_names_full, _ = create_dataframe(args.data_dir)
    features, labels = create_tensor(df, image_names, painter_names_full, args.data_dir) 
    train_loader, val_loader, _ = get_dataloader(features, labels, args.batch_size)
   
    # Create model
    model = create_model("resnet18", df)

    # Check imbalance and calculate class weights
    weights =  torch.tensor(calculate_class_weights(train_loader.dataset.tensors[1], 10)).to(args.device)


    criterion = torch.nn.CrossEntropyLoss(weights = weights)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    # Train the model
    train_model(model, args.device, train_loader, val_loader, criterion, optimizer, num_epochs=args.epochs)

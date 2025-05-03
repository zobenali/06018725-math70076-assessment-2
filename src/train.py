import numpy as np
import tqdm
import torch

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

    n_epoch = 40

    for epoch in range(n_epoch):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epoch}', unit='batch')

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
        print(f"Epoch {epoch+1}/{n_epoch}, Train Loss: {running_loss/len(train_loader.dataset):.4f}, Val Loss: {val_loss/len(val_loader.dataset):.4f}, Val Acc: {(correct/total)*100:.2f}%")

    torch.save(model.state_dict(), 'model.pth')
    
    return model, train_L, val_L

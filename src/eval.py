import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data_processing import create_dataframe, create_tensor, get_dataloader
from model import create_model 


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model: The PyTorch model to evaluate.
        test_loader: DataLoader for the test dataset.
        criterion: Loss function.
        device: Device to run the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
        A dictionary containing test loss, test accuracy, predictions, and targets.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc='Validating', unit='batch')
        for inputs, targets in test_loader_tqdm:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        test_loader_tqdm.set_postfix(test_loss=test_loss / len(test_loader.dataset), test_acc=(correct / total) * 100)

    print(f'Test Loss: {test_loss / len(test_loader.dataset):.4f}, Test Accuracy: {100. * correct / total:.2f}%')

    return test_loss / len(test_loader.dataset), correct / total

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a PyTorch model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation.")
    args = parser.parse_args()

    # Load data
    df, image_names, _, painter_names_full, _ = create_dataframe(args.data_dir)
    features, labels = create_tensor(df, image_names, painter_names_full, args.data_dir) 
    _, _, test_loader = get_dataloader(features, labels, args.batch_size)
   

    model = create_model('resnet18', df )  
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss() 

    # Evaluate the model
    results = evaluate_model(model, test_loader, criterion, args.device)

    # Print results
    print("Evaluation Results:")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
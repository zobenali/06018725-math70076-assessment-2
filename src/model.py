import torch
import torch.nn as nn
from torchvision import models
from data_processing import get_num_classes


def create_model(model_name, dataframe):
    num_classes = get_num_classes(dataframe)
    if model_name == 'resnet18':
        model = models.resnet18(weights = models.ResNet18_Weights) # Use the latest weights available
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    if hasattr(model, 'fc')  : 
        n_features = model.fc.in_features
        model.fc = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(n_features, 256),
                                nn.ReLU(),
                                #nn.Dropout(p=0.5),
                                nn.Linear(256, num_classes)
                            )
        
    
    else : 
        n_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(n_features, 256),
                                nn.ReLU(),
                                #nn.Dropout(p=0.5),
                                nn.Linear(256, num_classes)
                            )
    
    return model

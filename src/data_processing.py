import os
import glob
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


normalise_resize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalisation
])

def format_image(img_file):
	"""
	This function reads and formats an image so that it can be fed to the ResNET network

	Parameters
	----------
	img_file : image file name

	Returns
	-------
	img_out_model : the correctly formatted image 
	img : the image as read by the load_img function of keras.preprocessing.image
	"""
	# Read image
	img = Image.open(img_file)
	img_tensor = normalise_resize(img).unsqueeze(axis=0)
	img_np = np.array(img)

	return img_tensor, img_np

def unformat_image(img_in):
	"""
	This function inverts the preprocessing applied to images for use in the VGG16/inceptionv3 network

	Parameters
	----------
	img_file : formatted image of shape (batch_size,m,n,3)

	Returns
	-------
	img_out : a m-by-n-by-3 array, representing an image that can be written to an image file
	"""

	img_in = np.transpose(img_in.detach().numpy().squeeze(),[1,2,0])
	# invert the mean and standard deviation
	mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
	std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
	img_out = std * img_in + mean
	img_out *= 255
	img_out = np.uint8(np.clip(img_out, 0, 255))

	return img_out

def create_dataframe(working_directory):
    painters = [name for name in os.listdir(working_directory) if os.path.isdir(os.path.join(working_directory, name))] #liste des peintres

    images_by_painter = {} # Dictionnary with painter name as key and list of images as value

    image_names = []
    painter_names = []
    painter_names_full = []
    bdates = []

    for painter in painters:
        painter_path = os.path.join(working_directory, painter)
        image_paths = glob.glob(os.path.join(painter_path, '*'))
        images_by_painter[painter] = image_paths
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            image_names.append(image_name)
            painter_names.append(painter.split(' - ')[0]) # Only keep the name of the painter, not the birthdate
            painter_names_full.append(painter)
            bdates.append(int(painter.split('_')[1])) # Keep the birthdate as an integer

    df = pd.DataFrame({
        'image_name': image_names,
        'painter': painter_names,
        'date' : bdates
    })

    return df,image_names, painter_names, painter_names_full, images_by_painter

def create_tensor(df, image_names, painter_names_full, Working_directory):
    """
    This function creates a tensor of images and a tensor of labels from the dataframe
    and the image names
    Parameters
    ---------- 
    df : pandas dataframe
        dataframe containing the image names and the labels 
    image_names : list of str
        list of image names
    painter_names_full : list of str
        list of painter names
    Working_directory : str
        path to the working directory  
    
    Returns
    ------- 
    X_tensor : torch.Tensor 
        tensor of images of shape (n_images, 3, 224, 224)
    Y_tensor : torch.Tensor
        tensor of labels of shape (n_images,)   
    
    """
    X = [] #Image Tensor
    Y = [] #labels

    label_to_int = {label: idx for idx, label in enumerate(df['painter'].unique())} #change painter as

    for n in range(len(image_names)):
        img_path = os.path.join(Working_directory, painter_names_full[n], df.iloc[n]['image_name'])
        img_tensor, _ = format_image(img_path)

        X.append(img_tensor)
        Y.append(label_to_int[df.iloc[n]['painter']])
    
    X_tensor = torch.cat(X, dim=0)
    Y_tensor = torch.tensor(Y, dtype=torch.long)

    return X_tensor, Y_tensor

def get_num_classes(df):
    """
    This function returns the number of classes in the dataframe
    Parameters
    ---------- 
    df : pandas dataframe
        dataframe containing the image names and the labels 
    
    Returns
    ------- 
    num_classes : int 
        number of classes in the dataframe
    
    """
    num_classes = len(df['painter'].unique())

    return num_classes

def get_dataloader(feature_tensor, label_tensor, batch_size=32):
    """
    This function creates a dataloader from the feature tensor and the label tensor
    Parameters
    ---------- 
    feature_tensor : torch.Tensor
        tensor of images of shape (n_images, 3, 224, 224)
    label_tensor : torch.Tensor
        tensor of labels of shape (n_images,) 
    batch_size : int
        size of the batches
    
    Returns
    ------- 
    dataloader : torch.utils.data.DataLoader 
        dataloader for the training set
    
    """
    X_train, X_test, Y_train, Y_test = train_test_split(feature_tensor,label_tensor, test_size=0.15)
    x_test, x_val, y_test, y_val = train_test_split(X_test, Y_test, test_size=0.5)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)


    train_loader =torch.utils.data. DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader
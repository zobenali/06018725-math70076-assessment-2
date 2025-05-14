import argparse
import numpy as np
import torch
import pickle
import pandas as pd
from PIL import Image
from scipy.stats import chi2
from data_processing import format_image
from gmm_analysis import run_gmm, extract_features, add_birthdate_feature, run_tsne
from model import create_model


def calculate_joint_probabilities(X_new, X_train, tsne_model):
    """
    This function aims to compute joint probability between new vector of features and the model ones.
    Parameters
    ----------
    X_new : vector of features
    X_train : inputs of the model
    tsne_model : used model

    Returns
    -------
    P_joint : joint_probability
    """
    perplexity = tsne_model.perplexity
    n_samples_new = X_new.shape[0]
    n_samples_train = X_train.shape[0]

    P_joint = np.zeros((n_samples_new, n_samples_train))

    for i in range(n_samples_new):
        x_new_i = X_new[i]
        # Compute Euclidean distances
        distances = np.sum((X_train - x_new_i) ** 2, axis=1)

        # Compute conditional probabilities
        pij = np.zeros(n_samples_train)
        sum_pi = 0.0
        for j in range(n_samples_train):
            if i != j:
                pij[j] = np.exp(-distances[j] / (2 * (perplexity * tsne_model.early_exaggeration) ** 2))
                sum_pi += pij[j]

        # Normalize
        P_joint[i, :] = pij / sum_pi

    return P_joint

def calculate_tsne_point(X_train, X_new, tsne_model):
    """
    This function returns the coordinates of the new canva which features are X_new in TSNE map
    ----------
    X_new : vector of features
    X_train : inputs of the model
    tsne_model : used model

    Returns
    -------
    tsne_coordinates : coordinates of X_new in TSNE MAP
    """
    P = tsne_model.kl_divergence_
    Q = tsne_model.embedding_

    P_new = calculate_joint_probabilities(X_new,X_train,tsne_model)
    distances = np.sum(P_new * np.log((P_new + 1e-15) / (P + 1e-15)), axis=0)

    tsne_coordinates = np.dot(distances, Q)

    return tsne_coordinates

def mahalanobis_distance(point, mean, covariance):
    diff = point - mean
    inv_cov = np.linalg.inv(covariance)
    return np.sqrt(diff.T @ inv_cov @ diff)

def euclidean_distance(point, mean):
    return np.sqrt(np.sum((point - mean) ** 2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the art movement of a painting using t-SNE and GMM")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tensor_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--csv_path", required=True, help="CSV file with columns: image_name,painter,date,movement")
    parser.add_argument("--birthdate", required=True, help="Birthyeear of the artist")
    parser.add_argument("--device", default="cpu", help="Device to use for computation (cpu or cuda)")
    parser.add_argument("--n_components", type=int, default=13, help="Number of GMM components (default: 4)")
    args = parser.parse_args()


    # Load CSV file
    df = pd.read_csv(args.csv_path)
    print(len(df['painter'].unique()))
    # Load model
    model = create_model("resnet18", df)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    print(f"Model loaded from {args.model_path}")
    # Load tensors
    with open(args.tensor_path, "rb") as f:
        data = pickle.load(f)
        X_tensor = data["X"]
        Y_tensor = data["Y"]
    
    print(f"Tensors loaded from {args.tensor_path}")


    features, labels = extract_features(model, X_tensor, Y_tensor, args.device)
    birthdates = df['date']

    print(f"Features and labels extracted from tensors")
    features_b = add_birthdate_feature(features, birthdates)

    num_classes = len(df['painter'].unique())
    movements = df['movement'].to_numpy()
    all_movements = sorted(df['movement'].unique())
    label_to_index = {label: idx for idx, label in enumerate(all_movements)}

    tsne, features_tsne = run_tsne(features_b)
    print(f"t-SNE computed")

    # Get coordinates of the new point 
    image, _ = format_image(args.image_path)

    features_img = []
    labels_img = []
    model.eval()

    with torch.no_grad():
        img_tensor = image.to(args.device)
        outputs = model(img_tensor)
        features_img.append(outputs)

    features_img = torch.cat(features_img)
    features_img = features_img.cpu().numpy()

    features_with_birth_img = np.zeros((1, num_classes + 1))

    for j in range(num_classes) :
        features_with_birth_img[0,j] = features_img[0,j]
        features_with_birth_img[0,-1] = args.birthdate
    
    X_new_tsne = calculate_tsne_point(features_b, features_with_birth_img, tsne)
    print(f"t-SNE coordinates of the new point computed")
    # GMM
    gmm,_  = run_gmm(features_tsne, n_components=args.n_components)
    means = gmm.means_
    covariances = gmm.covariances_

    # Check Mahalanobis distance to ellipses
    threshold = chi2.ppf(0.95, df=2)
    point_within = []

    for i, (mean, cov) in enumerate(zip(means, covariances)):
        dist = mahalanobis_distance(X_new_tsne, mean, cov)
        if dist <= np.sqrt(threshold):
            point_within.append(all_movements[i])

    if point_within:
        print("The painting might belong to the following movements:")
        for m in point_within:
            print(f"- {m}")
    else:
        print("The painting doesn't strongly belong to any known movement based on the model.")

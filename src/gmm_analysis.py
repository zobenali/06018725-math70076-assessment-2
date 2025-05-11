# gmm_analysis.py

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def extract_features(model, X_tensor, Y_tensor, device):
    model.eval()
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            features.append(outputs)
            labels.extend(targets)
    features = torch.cat(features).cpu().numpy()
    labels = torch.tensor(labels).numpy()
    return features, labels

def add_birthdate_feature(features, birthdates):
    features_with_date = np.zeros((features.shape[0], features.shape[1] + 1))
    features_with_date[:, :-1] = features
    features_with_date[:, -1] = birthdates / 10
    return features_with_date

def run_tsne(features):
    tsne = TSNE(n_components=2, random_state=1)
    return tsne.fit_transform(features)

def run_gmm(tsne_features, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=1)
    gmm.fit(tsne_features)
    labels = gmm.predict(tsne_features)
    return gmm, labels

def compute_centroids(tsne_features, labels, num_classes):
    x_moy = np.zeros(num_classes)
    y_moy = np.zeros(num_classes)
    counts = np.zeros(num_classes)
    for i, label in enumerate(labels):
        x_moy[label] += tsne_features[i, 0]
        y_moy[label] += tsne_features[i, 1]
        counts[label] += 1
    x_moy /= counts
    y_moy /= counts
    return np.vstack((x_moy, y_moy)).T

def plot_gaussian_ellipse(mean, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)

def plot_tsne_with_gmm(tsne_features, labels, gmm, label_colors, label_names, title):
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.scatter(tsne_features[:, 0], tsne_features[:, 1], color=label_colors[labels], marker='o', alpha=0.6)
    ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='black', marker='x')

    for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
        plot_gaussian_ellipse(mean, cov, ax, n_std=3, edgecolor='black')
        ax.text(mean[0], mean[1], str(i), fontsize=10, ha='right')

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[i], markersize=10, label=label_names[i])
                       for i in range(len(label_names))]

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

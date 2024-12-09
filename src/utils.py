import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler


EPOCHS = 20
BATCH_SIZE = 64
LATENT_DIMS = 10
LR = 0.001

CLASSES = [
    "Tops",
    "Trousers",
    "Pullovers",
    "Dresses",
    "Coats",
    "Sandals",
    "Shirts",
    "Sneakers",
    "Bags",
    "Ankle Boots",
]

NUM_CLASSES = len(CLASSES)


def latent_plot(model, data, title: str | None = None, latent_size=LATENT_DIMS):
    encodings, labels = create_encodings(model, data, latent_size)
    scaler = StandardScaler()
    scaled_encodings = scaler.fit_transform(encodings)

    pca = PCA(n_components=2)
    encodings_pca = pca.fit_transform(scaled_encodings)

    for i in range(10):
        mask = labels == i
        plt.scatter(
            encodings_pca[mask, 0],
            encodings_pca[mask, 1],
            s=0.25,
            label=CLASSES[i],
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(
        "Latent Space Plot using Principle Component Analysis"
        if title is None
        else title
    )
    plt.legend()
    plt.show()


def create_encodings(model, data, latent_size=LATENT_DIMS):
    n_data = len(data.dataset)

    encodings = []
    labels = []

    for idx, (image, label) in enumerate(data):
        image = image.to(device)
        latent = model.encode(image)
        encodings.extend(latent.cpu().detach().numpy())
        labels.extend(label)
    labels = np.array(labels)
    encodings = np.array(encodings)
    encodings = encodings.reshape((n_data, latent_size))
    return encodings, labels


def get_linear_classification_acc(model, train, test, latent_size=LATENT_DIMS):
    lr_model = LogisticRegression(max_iter=1000)
    train_encodings, train_labels = create_encodings(model, train, latent_size)
    test_encodings, test_labels = create_encodings(model, test, latent_size)
    lr_model.fit(train_encodings, train_labels)

    pred_labels = lr_model.predict(test_encodings)
    return accuracy_score(test_labels, pred_labels)


mpl.rcParams["figure.dpi"] = 300

base_transform = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.2864], std=[0.3203]),
]

augment_transform = [
    # Rotation ±10 degrees
    v2.RandomApply(transforms=torch.nn.ModuleList([v2.RandomRotation(10)])),
    # Scale ±20%
    v2.RandomApply(
        transforms=torch.nn.ModuleList([v2.RandomAffine(scale=(0.8, 1.2), degrees=0)])
    ),
    # Translation ±4px
    v2.RandomApply(
        transforms=torch.nn.ModuleList(
            [v2.RandomAffine(translate=(1 / 7, 1 / 7), degrees=0)]
        )
    ),
]

train_base = DataLoader(
    FashionMNIST(
        root="/Users/henrywilliams/Documents/uni/amml/lab5/src/data",
        train=True,
        transform=v2.Compose(base_transform),
    ),
    shuffle=False,
    batch_size=64,
)

test_base = DataLoader(
    FashionMNIST(
        root="/Users/henrywilliams/Documents/uni/amml/lab5/src/data",
        train=False,
        transform=v2.Compose(base_transform),
    ),
    shuffle=False,
    batch_size=64,
)

train_augment = DataLoader(
    FashionMNIST(
        root="/Users/henrywilliams/Documents/uni/amml/lab5/src/data",
        train=True,
        transform=v2.Compose(base_transform + augment_transform),
    ),
    shuffle=False,
    batch_size=64,
)

test_augment = DataLoader(
    FashionMNIST(
        root="/Users/henrywilliams/Documents/uni/amml/lab5/src/data",
        train=False,
        transform=v2.Compose(base_transform + augment_transform),
    ),
    shuffle=False,
    batch_size=64,
)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

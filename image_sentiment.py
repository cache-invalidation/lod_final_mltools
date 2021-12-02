import os

import numpy as np

import torch
import torchvision.transforms as t

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader

from .sentiment import Sentiment
from .vgg19 import KitModel as VGG19

class ImagesDataset (Dataset):
    def __init__(self, filenames, root=None, transform=None):
        super(ImagesDataset).__init__()

        self.list = filenames
        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        path = self.list[index]

        if self.root is not None:
            path = os.path.join(self.root, path)

        x = default_loader(path)
        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.list)

transform = t.Compose([
    t.Resize((224, 224)),
    t.ToTensor(),
    t.Lambda(lambda x: x[[2, 1, 0], ...] * 255),
    t.Normalize(mean=[116.8007, 121.2751, 130.4602], std=[1,1,1]),
])

# Load model and switch it into the evaluation mode
model = VGG19
model = model(os.path.join(os.path.dirname(__file__), 'image_models/vgg19_finetuned_all.pth'))
model.eval()

def make_predictions(images: list[str], batch_size=64, num_workers=8) -> list[Sentiment]:
    """
    Predict sentiment for images

    Arguments:
    ----------
        images: list[str]
            List of paths to images to predict sentiment for

        batch_size: int
            Batch size for model running

    Returns:
    --------
        list[Sentiment]
            Predictions for provided images
    """

    predicted = np.array([Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE])

    data = ImagesDataset(images, transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers)

    result = list()

    with torch.no_grad():
        for x in dataloader:
            pred = model(x).numpy()
            classes = predicted[np.argmax(pred, axis=1)]
            result += list(classes)

    return result

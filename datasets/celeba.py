import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


# A very simplistic implementation of the CelebA dataset supporting only images and no annotations
class CelebADataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
    ):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform

        self.images = []

        for img in os.listdir(root):
            self.images.append(os.path.join(self.root, img))

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)

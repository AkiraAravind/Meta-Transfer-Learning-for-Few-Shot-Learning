import os, random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class MiniImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, split)
        self.classes = sorted(os.listdir(self.root))
        self.samples = []
        for c in self.classes:
            cpath = os.path.join(self.root, c)
            for f in os.listdir(cpath):
                self.samples.append((os.path.join(cpath, f), c))
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.transform = transform or T.Compose([
            T.Resize((84,84)),
            T.ToTensor()
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), self.class_to_idx[cls]

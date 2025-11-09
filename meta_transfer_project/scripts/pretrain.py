# trains a linear classifier on top of ResNet12 features to get backbone weights
import argparse, torch, os
from torch.utils.data import DataLoader
from models.resnet12 import ResNet12
from datasets.mini_imagenet import MiniImageNetDataset
import torch.nn as nn, torch.optim as optim, tqdm

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = ResNet12(base_channels=32, channels_mult=args.channels_mult).to(device)
    train_ds = MiniImageNetDataset(args.data_root, 'train')
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    classifier = nn.Linear(backbone.out_dim, len(train_ds.classes)).to(device)
    opt = optim.SGD(list(backbone.parameters()) + list(classifier.parameters()), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        backbone.train(); classifier.train()
        pbar = tqdm.tqdm(loader)
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            feats = backbone(x)
            logits = classifier(feats)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_description(f"Epoch {epoch} loss {loss.item():.4f}")
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save({'backbone': backbone.state_dict(), 'classifier': classifier.state_dict()}, args.ckpt)
    print("Saved:", args.ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/miniImageNet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--channels_mult', type=float, default=1.0)
    parser.add_argument('--ckpt', default='checkpoints/backbone_pretrain.pth')
    main(parser.parse_args())

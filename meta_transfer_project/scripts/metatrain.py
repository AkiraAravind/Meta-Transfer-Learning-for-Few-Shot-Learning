# # episodic meta-train with SS parameters
# import argparse, os, random, torch, tqdm
# from torch.utils.data import DataLoader
# from models.resnet12 import ResNet12
# from models.ss_layers import ScaleShiftWrapper
# from datasets.mini_imagenet import MiniImageNetDataset
# import torch.nn as nn, torch.optim as optim
# import numpy as np

# # helper to wrap conv/linear layers with SS
# def wrap_backbone(backbone):
#     for name, module in backbone.named_modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             parent = backbone
#             # replace at attribute path
#             path = name.split('.')
#             # find parent and attribute name
#             ptr = backbone
#             for p in path[:-1]:
#                 ptr = getattr(ptr, p)
#             setattr(ptr, path[-1], ScaleShiftWrapper(module))
#     return backbone

# def sample_episode(data_root, split, way, shot, query):
#     # simple episode sampler returning tensors; for brevity the actual implementation is left to you
#     raise NotImplementedError("Use a simple sampler that picks `way` classes and samples shot+query images each.")

# def main(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # load pretrain checkpoint
#     backbone = ResNet12(base_channels=32, channels_mult=args.channels_mult)
#     ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')
#     backbone.load_state_dict(ckpt['backbone'])
#     # wrap conv/linear for SS
#     backbone = wrap_backbone(backbone).to(device)
#     # freeze base weights (they're buffers) and only keep alpha/beta params trainable
#     meta_params = [p for n,p in backbone.named_parameters() if 'alpha' in n or 'beta' in n]
#     opt = optim.Adam(meta_params, lr=args.meta_lr)
#     # episodic loop (simplified)
#     for epoch in range(args.meta_epochs):
#         backbone.train()
#         # sample a batch of episodes (meta-batch)
#         for meta_iter in range(args.meta_iters_per_epoch):
#             # build a batch of episodes and compute meta-loss, then update meta-params
#             # (implementation detail omitted: you will create per-episode support/query forward passes)
#             pass
#     torch.save({'mtl': backbone.state_dict()}, args.out_ckpt)
#     print("Saved meta checkpoint:", args.out_ckpt)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pretrain_ckpt', default='checkpoints/backbone_pretrain.pth')
#     parser.add_argument('--out_ckpt', default='checkpoints/mtl_meta.pth')
#     parser.add_argument('--channels_mult', type=float, default=1.0)
#     parser.add_argument('--meta_lr', type=float, default=1e-3)
#     parser.add_argument('--meta_epochs', type=int, default=40)
#     parser.add_argument('--meta_iters_per_epoch', type=int, default=50)
#     main(parser.parse_args())





import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.resnet12 import ResNet12
from models.ss_layers import ScaleShiftWrapper
from datasets.mini_imagenet import MiniImageNetDataset
from torchvision import transforms
from PIL import Image
import random
import os

# --------------------------
# Utility: wrap backbone with Scale & Shift layers
# --------------------------
def wrap_backbone(backbone):
    for name, module in backbone.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            ptr = backbone
            path = name.split('.')
            for p in path[:-1]:
                ptr = getattr(ptr, p)
            setattr(ptr, path[-1], ScaleShiftWrapper(module))
    return backbone


# --------------------------
# Episode sampling function
# --------------------------
def sample_episode(dataset, split, way=5, shot=1, query=15):
    classes = random.sample(dataset.classes, way)
    support_x, support_y, query_x, query_y = [], [], [], []
    label_map = {c: i for i, c in enumerate(classes)}

    for c in classes:
        imgs = [p for p, lbl in dataset.samples if lbl == c]
        sampled = random.sample(imgs, shot + query)
        sup_imgs, qry_imgs = sampled[:shot], sampled[shot:]

        for p in sup_imgs:
            img = dataset.transform(Image.open(p).convert("RGB"))
            support_x.append(img)
            support_y.append(label_map[c])
        for p in qry_imgs:
            img = dataset.transform(Image.open(p).convert("RGB"))
            query_x.append(img)
            query_y.append(label_map[c])

    return (torch.stack(support_x), torch.tensor(support_y),
            torch.stack(query_x), torch.tensor(query_y))


# --------------------------
# Episode loss evaluation
# --------------------------
def evaluate_episode_loss(backbone, s_x, s_y, q_x, q_y, device, grad=False):
    backbone.train(grad)
    s_x, s_y, q_x, q_y = s_x.to(device), s_y.to(device), q_x.to(device), q_y.to(device)

    supp_feats = backbone(s_x)
    qry_feats = backbone(q_x)

    # Prototype computation
    n_way = len(torch.unique(s_y))
    prototypes = []
    for i in range(n_way):
        prototypes.append(supp_feats[s_y == i].mean(0))
    prototypes = torch.stack(prototypes)

    qry_norm = F.normalize(qry_feats, dim=1)
    proto_norm = F.normalize(prototypes, dim=1)
    logits = qry_norm @ proto_norm.t() * 10
    loss = F.cross_entropy(logits, q_y)

    return loss if grad else loss.item()


# --------------------------
# Main meta-training function
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_ckpt", type=str, required=True, help="path to pretrained backbone")
    parser.add_argument("--out_ckpt", type=str, default="checkpoints/mtl_meta.pth", help="path to save meta model")
    parser.add_argument("--channels_mult", type=float, default=0.75)
    parser.add_argument("--meta_lr", type=float, default=1e-3)
    parser.add_argument("--meta_epochs", type=int, default=60)
    parser.add_argument("--meta_iters_per_epoch", type=int, default=50)
    parser.add_argument("--data_root", type=str, default="data/miniImageNet")

    # Hard Task meta-batch arguments
    parser.add_argument("--ht_on", action="store_true", help="Enable Hard Task Meta-Batch")
    parser.add_argument("--ht_pool", type=int, default=20, help="Number of candidate episodes in the pool")
    parser.add_argument("--meta_batch", type=int, default=4, help="Number of hardest episodes to use for update")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running meta-train on {device} | HT mode: {args.ht_on}")

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])
    train_ds = MiniImageNetDataset(args.data_root, "train", transform=transform)

    # Load pretrained backbone
    backbone = ResNet12(base_channels=32, channels_mult=args.channels_mult)
    backbone = wrap_backbone(backbone).to(device)
    ckpt = torch.load(args.pretrain_ckpt, map_location=device)
    if "backbone" in ckpt:
        backbone.load_state_dict(ckpt["backbone"], strict=False)
    else:
        backbone.load_state_dict(ckpt, strict=False)

    # Meta optimizer
    opt = torch.optim.Adam(
        [p for p in backbone.parameters() if p.requires_grad],
        lr=args.meta_lr
    )

    # Meta-training loop
    for epoch in range(args.meta_epochs):
        losses = []
        for meta_iter in tqdm(range(args.meta_iters_per_epoch), desc=f"Epoch {epoch}"):
            if args.ht_on:
                # Step 1: Sample candidate pool
                pool_losses, pool_data = [], []
                for _ in range(args.ht_pool):
                    s_x, s_y, q_x, q_y = sample_episode(train_ds, "train", way=5, shot=1, query=15)
                    loss_val = evaluate_episode_loss(backbone, s_x, s_y, q_x, q_y, device, grad=False)
                    pool_losses.append(loss_val)
                    pool_data.append((s_x, s_y, q_x, q_y))

                # Step 2: Select hardest episodes
                hard_indices = np.argsort(pool_losses)[-args.meta_batch:]
                hard_episodes = [pool_data[i] for i in hard_indices]
            else:
                # Random episodes
                hard_episodes = [sample_episode(train_ds, "train", way=5, shot=1, query=15)
                                 for _ in range(args.meta_batch)]

            # Step 3: Compute gradients on selected hard tasks
            total_loss = 0.0
            for (s_x, s_y, q_x, q_y) in hard_episodes:
                loss = evaluate_episode_loss(backbone, s_x, s_y, q_x, q_y, device, grad=True)
                total_loss += loss
            total_loss /= len(hard_episodes)

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            losses.append(total_loss.item())

        print(f"Epoch {epoch} mean loss: {np.mean(losses):.4f}")

    # Save meta-trained model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"mtl": backbone.state_dict()}, args.out_ckpt)
    print(f"✅ Saved meta checkpoint: {args.out_ckpt}")


if __name__ == "__main__":
    main()

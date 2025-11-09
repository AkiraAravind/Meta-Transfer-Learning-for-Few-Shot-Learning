import torch, argparse, numpy as np, random, os, json
from models.resnet12 import ResNet12
from models.ss_layers import ScaleShiftWrapper
from datasets.mini_imagenet import MiniImageNetDataset
from torchvision import transforms
from scipy import stats

# -----------------------------
# Helper Functions
# -----------------------------
def wrap_backbone(backbone):
    for name, module in backbone.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            ptr = backbone
            path = name.split('.')
            for p in path[:-1]:
                ptr = getattr(ptr, p)
            setattr(ptr, path[-1], ScaleShiftWrapper(module))
    return backbone

def compute_confidence_interval(results):
    arr = np.array(results)
    mean = arr.mean()
    se = arr.std(ddof=1)/np.sqrt(len(arr))
    ci95 = 1.96 * se
    return mean, ci95

def sample_episode(dataset, way, shot, query):
    """Returns tensors for support/query sets for a single episode"""
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

def evaluate_episode(backbone, support_x, support_y, query_x, query_y, device):
    backbone.eval()
    with torch.no_grad():
        supp_feats = backbone(support_x.to(device))
        qry_feats = backbone(query_x.to(device))
    # compute prototypes (mean feature per class)
    n_way = torch.unique(support_y).size(0)
    prototypes = []
    for i in range(n_way):
        prototypes.append(supp_feats[support_y == i].mean(0))
    prototypes = torch.stack(prototypes)  # [way, feat_dim]
    # cosine similarity classifier
    qry_norm = torch.nn.functional.normalize(qry_feats, dim=1)
    proto_norm = torch.nn.functional.normalize(prototypes, dim=1)
    logits = qry_norm @ proto_norm.t() * 10  # temperature
    preds = torch.argmax(logits, dim=1)
    acc = (preds.cpu() == query_y).float().mean().item()
    return acc

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/mtl_meta.pth')
    parser.add_argument('--data_root', default='data/miniImageNet')
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--channels_mult', type=float, default=0.75)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running evaluation on", device)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])
    test_ds = MiniImageNetDataset(args.data_root, 'test', transform=transform)

    # Build model same as training
    backbone = ResNet12(base_channels=32, channels_mult=args.channels_mult)
    backbone = wrap_backbone(backbone).to(device)

    # Load checkpoint
    ck = torch.load(args.ckpt, map_location=device)
    backbone.load_state_dict(ck['mtl'], strict=False)
    print(f"✅ Loaded meta model from {args.ckpt}")

    # Run episodes
    results = []
    for epi in range(args.episodes):
        s_x, s_y, q_x, q_y = sample_episode(test_ds, args.way, args.shot, args.query)
        acc = evaluate_episode(backbone, s_x, s_y, q_x, q_y, device)
        results.append(acc)
        if (epi + 1) % 50 == 0:
            mean, ci95 = compute_confidence_interval(results)
            print(f"[{epi+1}/{args.episodes}] Mean Acc: {mean*100:.2f}% ± {ci95*100:.2f}%")

    mean, ci95 = compute_confidence_interval(results)
    print("=" * 60)
    print(f"Final Accuracy: {mean*100:.2f}% ± {ci95*100:.2f}% (95% CI)")
    print("=" * 60)

    # === Save results so plotting scripts can pick them up ===
    try:
        os.makedirs('results', exist_ok=True)
        out_path = os.path.join('results', 'eval_results.json')
        # load existing results if present, then update this shot
        existing = {}
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = {}

        key = f"{args.shot}-shot"
        existing[key] = {
            'mean_percent': round(mean * 100, 4),
            'ci95_percent': round(ci95 * 100, 4),
            'episodes': args.episodes
        }
        with open(out_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"✅ Saved evaluation results to {out_path} under key '{key}'")
    except Exception as e:
        print("⚠️  Failed to save evaluation results:", e)

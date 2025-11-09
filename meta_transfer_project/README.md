# Meta-Transfer Learning (MTL) — miniImageNet

**Objective:**  
Reproduce the Meta-Transfer Learning workflow (Yaoyao Liu et al.) using a lightweight PyTorch implementation on the miniImageNet dataset.

## Workflow
1. **Pretrain** ResNet-12 backbone (supervised on base classes).  
2. **Meta-Transfer Stage:** freeze base weights, learn scale (α) and shift (β) parameters via episodic few-shot tasks.  
3. **Evaluate** on unseen classes using 5-way 1-shot and 5-way 5-shot settings.

## Configuration
| Stage | Key Params | Epochs | Notes |
|-------|-------------|--------|------|
| Pretrain | lr = 0.01, momentum = 0.9 | 10 | feature extractor |
| Meta-Train | meta-lr = 1e-3, meta-batch = 4 | 30 | α/β updates only |
| Backbone | ResNet-12 (× 0.75 width) | — | 84×84 inputs |

## Results
| Setting | Accuracy (mean ± 95 % CI) |
|----------|---------------------------|
| 5-way 1-shot | **42.38 % ± 0.72 %** |
| 5-way 5-shot | **58.96 % ± 0.71 %** |

## Requirements



torch>=2.0
torchvision
numpy
Pillow
tqdm
matplotlib
scipy





## Run
```bash
# Pretrain
python scripts/pretrain.py --data_root data/miniImageNet --epochs 10 --batch 64 --channels_mult 0.75

# Meta-transfer
python scripts/metatrain.py --pretrain_ckpt checkpoints/backbone_pretrain.pth --out_ckpt checkpoints/mtl_meta.pth --meta_epochs 30 --channels_mult 0.75

# Evaluate
python scripts/evaluate.py --ckpt checkpoints/mtl_meta.pth --data_root data/miniImageNet --episodes 600 --way 5 --shot 1 --query 15 --channels_mult 0.75
python scripts/evaluate.py --ckpt checkpoints/mtl_meta.pth --data_root data/miniImageNet --episodes 600 --way 5 --shot 5 --query 15 --channels_mult 0.75





---

### 2️⃣ Slide Deck Outline (for your presentation)

**Slide 1 – Title**  
“Meta-Transfer Learning on miniImageNet”  – Akira A.

**Slide 2 – Objective**  
Show workflow & accuracy of MTL with ResNet-12 backbone.

**Slide 3 – Workflow Diagram**  
Dataset → Pretrain → Scale-Shift Wrap → Meta-Train → Evaluate (1-shot / 5-shot).

**Slide 4 – Model Overview**  
Explain `W' = α × W + β`, α & β learned in meta stage only.

**Slide 5 – Experimental Setup**  
miniImageNet 84×84, 5-way 1/5-shot, 600 episodes, ResNet-12 × 0.75.

**Slide 6 – Results**
| Setting | Accuracy |
|----------|-----------|
| 1-shot | 42.38 % ± 0.72 % |
| 5-shot | 58.96 % ± 0.71 % |

Include a bar chart or small line plot (see below).

**Slide 7 – Conclusion & Future Work**  
More epochs + HT meta-batch + full width ResNet-12 → expected ≈ 60 % / 75 % acc.  

---

### 3️⃣ Optional Plot Script
Create `scripts/plot_results.py`:

```python
import matplotlib.pyplot as plt

acc = [42.38, 58.96]
ci  = [0.72, 0.71]
labels = ['1-Shot', '5-Shot']

plt.bar(labels, acc, yerr=ci, capsize=6)
plt.ylabel('Accuracy (%)')
plt.title('MTL on miniImageNet (ResNet-12 × 0.75)')
plt.ylim(0,100)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('results/accuracy_bar.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt
import os, json

# === Your evaluated results ===
shots = ['1-Shot', '5-Shot']
acc = [42.38, 58.96]
ci = [0.72, 0.71]
try:
    path = os.path.join('results', 'eval_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        for i, k in enumerate(['1-shot', '5-shot']):
            if k in data:
                acc[i] = float(data[k].get('mean_percent', acc[i]))
                ci[i] = float(data[k].get('ci95_percent', ci[i]))
        print(f"Loaded plotting values from {path}")
    else:
        print(f"No {path} found — using fallback hard-coded values")
except Exception as e:
    print("Failed to load eval results, using fallback values:", e)

# === Output directory ===
os.makedirs('results', exist_ok=True)

# === Create plot ===
plt.figure(figsize=(6, 5))
colors = ['#7aa6f7', '#4f81bd']  # light → deep blue
bars = plt.bar(shots, acc, yerr=ci, capsize=5, color=colors, width=0.45)

# === Labels and style ===
plt.title('5-Way Few-Shot Classification (MTL on miniImageNet)', fontsize=13, pad=10)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Annotate values on bars
for bar, val, err in zip(bars, acc, ci):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
             f'{val:.2f} ± {err:.2f}%', ha='center', fontsize=10, color='black')

plt.tight_layout()

# === Save and show ===
out_path = os.path.join('results', 'mtl_1v5shot_accuracy.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Saved plot to {out_path}")

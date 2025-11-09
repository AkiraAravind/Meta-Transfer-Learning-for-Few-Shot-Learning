import matplotlib.pyplot as plt
import os, json

# === Try to read results from evaluation output (results/eval_results.json) ===
labels = ['1-Shot', '5-Shot']
acc = [42.38, 58.96]        # fallback mean accuracies (percent)
ci = [0.72, 0.71]           # fallback 95% CIs (percent)
try:
    path = os.path.join('results', 'eval_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        # look for keys '1-shot' and '5-shot'
        for i, k in enumerate(['1-shot', '5-shot']):
            if k in data:
                acc[i] = float(data[k].get('mean_percent', acc[i]))
                ci[i] = float(data[k].get('ci95_percent', ci[i]))
        print(f"Loaded plotting values from {path}")
    else:
        print(f"No {path} found — using fallback hard-coded values")
except Exception as e:
    print("Failed to load eval results, using fallback values:", e)

# === Create output directory ===
os.makedirs('results', exist_ok=True)

# === Plot setup ===
plt.figure(figsize=(6, 5))
bars = plt.bar(labels, acc, yerr=ci, capsize=6, width=0.5)

# Style & labels
plt.title('Meta-Transfer Learning (MTL) on miniImageNet', fontsize=13, pad=10)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add text on bars
for bar, val, err in zip(bars, acc, ci):
    plt.text(bar.get_x() + bar.get_width()/2, val + 1.5,
             f'{val:.2f} ± {err:.2f}%', ha='center', fontsize=10)

plt.tight_layout()

# === Save and show ===
out_path = os.path.join('results', 'accuracy_bar.png')
plt.savefig(out_path, dpi=300)
plt.show()

print(f"✅ Plot saved to {out_path}")

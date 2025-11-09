import matplotlib.pyplot as plt
import os, json

labels = ['1-Shot', '5-Shot']
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

os.makedirs('results', exist_ok=True)

plt.figure(figsize=(5.8, 4.5))
colors = ['#b388eb', '#7e57c2']  # lavender → violet
bars = plt.bar(labels, acc, yerr=ci, capsize=5, width=0.45, color=colors)

plt.title('Meta-Transfer Learning (MTL) on miniImageNet', fontsize=13, pad=10)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar, val, err in zip(bars, acc, ci):
    plt.text(bar.get_x() + bar.get_width()/2, val + 1.5,
             f'{val:.2f} ± {err:.2f}%', ha='center', fontsize=10)

plt.tight_layout()
out_path = os.path.join('results', 'accuracy_bar_pretty.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Saved nicer plot to {out_path}")

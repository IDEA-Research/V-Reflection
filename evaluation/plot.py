import os
import re
import glob
import json
import matplotlib.pyplot as plt

# 结果目录：以脚本所在目录为基准，兼容从任意目录运行
_script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(_script_dir, "/comp_robot/zhoujiazhou/projects/Active-Coconut/evaluation/results/blink/decoding_by_steps/dit_recon/SFT_dit_recon_steps2500_b4_dit1.0_resampler0.1_acc8")

steps = [4, 8, 12]
pattern = re.compile(r"ck-(\d+)-step(\d+)\.json")

step_data = {s: [] for s in steps}

for step in steps:
    for path in glob.glob(os.path.join(result_dir, f"ck-*-step{step}.json")):
        name = os.path.basename(path)
        m = pattern.match(name)
        if not m:
            continue
        ck = int(m.group(1))
        with open(path, "r") as f:
            data = json.load(f)
        overall_acc = data[0]["overall_accuracy"]
        step_data[step].append((ck, overall_acc))
    step_data[step].sort(key=lambda x: x[0])

if not any(step_data[s] for s in steps):
    raise SystemExit(f"未找到任何结果文件，请检查路径: {result_dir}")

plt.figure(figsize=(8, 5))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
markers = ["o", "s", "^"]

for i, step in enumerate(steps):
    if not step_data[step]:
        continue
    cks = [x[0] for x in step_data[step]]
    accs = [x[1] for x in step_data[step]]
    plt.plot(cks, accs, marker=markers[i], color=colors[i], label=f"step {step}")
    for ck, acc in zip(cks, accs):
        plt.annotate(f"{acc:.2f}", (ck, acc), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7, color=colors[i])

plt.xlabel("ck number")
plt.ylabel("overall accuracy")
plt.title("SFT_box_resampler accuracy by step (BLINK)")
plt.legend()
plt.grid(True)
all_cks = sorted(set(ck for s in steps for ck, _ in step_data[s]))
plt.xticks(all_cks)
plt.tight_layout()
out_path = os.path.join(_script_dir, "step4_8_12_BLINK.png")
plt.savefig(out_path, dpi=200)
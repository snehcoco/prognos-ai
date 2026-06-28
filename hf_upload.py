"""
hf_upload.py — Upload all model checkpoints to HuggingFace Hub.
Run once from your terminal:
    python hf_upload.py YOUR_HF_TOKEN
"""
import sys, os

TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HF_TOKEN", "")
if not TOKEN:
    print("Usage: python hf_upload.py YOUR_HF_TOKEN")
    sys.exit(1)

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Run: pip install huggingface_hub")
    sys.exit(1)

api     = HfApi(token=TOKEN)
REPO_ID = "subhasreeee/prognos-ai-models"
BASE    = os.path.dirname(os.path.abspath(__file__))

def find(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

files = [
    # ISIC 2018 — from ISIC2018_classification_outputs\checkpoints
    (find([
        r"C:\Users\MOUSHIMI\Downloads\ISIC2018_classification_outputs\checkpoints\efficientnet_b0_best.pth",
     ]), "ISIC2018/efficientnet_b0_best.pth"),
    (find([
        r"C:\Users\MOUSHIMI\Downloads\ISIC2018_classification_outputs\checkpoints\resnet50_best.pth",
     ]), "ISIC2018/resnet50_best.pth"),
    (find([
        r"C:\Users\MOUSHIMI\Downloads\ISIC2018_classification_outputs\checkpoints\densenet121_best.pth",
     ]), "ISIC2018/densenet121_best.pth"),

    # ISIC 2019 — from streamlit_app\checkpoints\ISIC2019
    (find([os.path.join(BASE, "checkpoints", "ISIC2019", "efficientnet_b0_best.pth")]),
     "ISIC2019/efficientnet_b0_best.pth"),
    (find([os.path.join(BASE, "checkpoints", "ISIC2019", "resnet50_best.pth")]),
     "ISIC2019/resnet50_best.pth"),
    (find([os.path.join(BASE, "checkpoints", "ISIC2019", "densenet121_best.pth")]),
     "ISIC2019/densenet121_best.pth"),

    # ISIC 2020 — check Downloads and checkpoints folder
    (find([
        r"C:\Users\MOUSHIMI\Downloads\best_efficientnet_b0_isic2020.pth",
        os.path.join(BASE, "checkpoints", "ISIC2020", "best_efficientnet_b0_isic2020.pth"),
     ]), "ISIC2020/best_efficientnet_b0_isic2020.pth"),
    (find([
        r"C:\Users\MOUSHIMI\Downloads\best_resnet50_isic2020.pth",
        os.path.join(BASE, "checkpoints", "ISIC2020", "best_resnet50_isic2020.pth"),
     ]), "ISIC2020/best_resnet50_isic2020.pth"),
    (find([
        r"C:\Users\MOUSHIMI\Downloads\best_densenet121_isic2020.pth",
        os.path.join(BASE, "checkpoints", "ISIC2020", "best_densenet121_isic2020.pth"),
     ]), "ISIC2020/best_densenet121_isic2020.pth"),

    # MILK10K — check Downloads and checkpoints folder
    (find([
        r"C:\Users\MOUSHIMI\Downloads\best_efficientnet_b0_milk10k.pth",
        os.path.join(BASE, "checkpoints", "MILK10K", "best_efficientnet_b0_milk10k.pth"),
     ]), "MILK10K/best_efficientnet_b0_milk10k.pth"),
    (find([
        r"C:\Users\MOUSHIMI\Downloads\best_resnet50_milk10k.pth",
        os.path.join(BASE, "checkpoints", "MILK10K", "best_resnet50_milk10k.pth"),
     ]), "MILK10K/best_resnet50_milk10k.pth"),
    (find([
        r"C:\Users\MOUSHIMI\Downloads\best_densenet121_milk10k.pth",
        os.path.join(BASE, "checkpoints", "MILK10K", "best_densenet121_milk10k.pth"),
     ]), "MILK10K/best_densenet121_milk10k.pth"),
]

print(f"Uploading to {REPO_ID}...\n")
skipped = 0
for i, (local, repo_path) in enumerate(files, 1):
    if not local:
        print(f"  [{i}/12] SKIPPED (not found): {repo_path}")
        skipped += 1
        continue
    mb = os.path.getsize(local) / 1024 / 1024
    print(f"  [{i}/12] {repo_path} ({mb:.0f} MB)...", end=" ", flush=True)
    api.upload_file(path_or_fileobj=local, path_in_repo=repo_path,
                    repo_id=REPO_ID, repo_type="model")
    print("✓")

print(f"\n{'✅ Done!' if not skipped else f'⚠️ Done with {skipped} skipped files.'}")
print(f"https://huggingface.co/{REPO_ID}")

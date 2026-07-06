"""
config.py — Central configuration for the Prognos AI Skin Cancer Detection app.
Model checkpoints are downloaded automatically from HuggingFace Hub on first run.
"""
import os

# ── Base directory (same folder as app.py) ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── HuggingFace repo that hosts all .pth checkpoints ─────────────────────────
HF_REPO_ID = "subhasreeee/prognos-ai-models"

# ── Local checkpoint folder ───────────────────────────────────────────────────
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ── Checkpoint file names per dataset (must match filenames on HuggingFace) ──
_CKPT_FILES = {
    "ISIC 2018": {
        "EfficientNet-B0": "ISIC2018/efficientnet_b0_best.pth",
        "ResNet50":        "ISIC2018/resnet50_best.pth",
        "DenseNet121":     "ISIC2018/densenet121_best.pth",
    },
    "ISIC 2019": {
        "EfficientNet-B0": "ISIC2019/efficientnet_b0_best.pth",
        "ResNet50":        "ISIC2019/resnet50_best.pth",
        "DenseNet121":     "ISIC2019/densenet121_best.pth",
    },
    "ISIC 2020": {
        "EfficientNet-B0": "ISIC2020/best_efficientnet_b0_isic2020.pth",
        "ResNet50":        "ISIC2020/best_resnet50_isic2020.pth",
        "DenseNet121":     "ISIC2020/best_densenet121_isic2020.pth",
    },
    "MILK10K": {
        "EfficientNet-B0": "MILK10K/best_efficientnet_b0_milk10k.pth",
        "ResNet50":        "MILK10K/best_resnet50_milk10k.pth",
        "DenseNet121":     "MILK10K/best_densenet121_milk10k.pth",
    },
}

# ── Build CKPT dict — local path for each checkpoint ─────────────────────────
CKPT = {
    ds: {
        arch: os.path.join(CKPT_DIR, rel)
        for arch, rel in models.items()
    }
    for ds, models in _CKPT_FILES.items()
}


def ensure_checkpoint(dataset: str, arch: str) -> tuple[str, str | None]:
    """
    Return (local_path, error_msg).
    Downloads from HuggingFace Hub if the file doesn't exist locally.
    """
    local_path = CKPT[dataset][arch]
    if os.path.isfile(local_path):
        return local_path, None

    try:
        from huggingface_hub import hf_hub_download
        hf_filename = _CKPT_FILES[dataset][arch]
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_filename,
            local_dir=CKPT_DIR,
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may nest the file — move it to the expected flat path
        if downloaded != local_path and os.path.isfile(downloaded):
            os.replace(downloaded, local_path)
        return local_path, None
    except Exception as e:
        return local_path, str(e)


# ── Plot image directories (bundled inside assets/ folder) ───────────────────
ASSETS = os.path.join(BASE_DIR, "assets")

PLOTS = {
    "ISIC 2018": os.path.join(ASSETS, "ISIC2018"),
    "ISIC 2019": os.path.join(ASSETS, "ISIC2019"),
    "ISIC 2020": os.path.join(ASSETS, "ISIC2020"),
    "MILK10K":   os.path.join(ASSETS, "MILK10K"),
}

# ── Dataset metadata ──────────────────────────────────────────────────────────
DATASETS = {
    "ISIC 2018": {
        "description": "Binary classification — Benign vs Malignant. 1,512 test images, 4.25:1 class imbalance.",
        "task": "binary",
        "num_classes": 2,
        "classes": ["Benign", "Malignant"],
        "metrics": {
            "EfficientNet-B0": {"Accuracy": 80.69, "Precision": 49.57, "Recall": 80.90, "F1": 61.48, "ROC-AUC": 0.9024},
            "ResNet50":        {"Accuracy": 82.80, "Precision": 53.21, "Recall": 80.56, "F1": 64.09, "ROC-AUC": 0.9024},
            "DenseNet121":     {"Accuracy": 76.85, "Precision": 44.71, "Recall": 90.97, "F1": 59.95, "ROC-AUC": 0.9188},
        },
        "plot_files": {
            "Class Distribution": "plots/class_distribution.png",
            "Sample Images":      "plots/sample_images.png",
            "Training Curves":    "plots/training_curves.png",
            "ROC Curves":         "plots/roc_curves.png",
            "Confusion Matrices": "plots/confusion_matrices.png",
            "Model Comparison":   "plots/model_comparison_bar_chart.png",
        },
        "gradcam_files": {
            "EfficientNet-B0": "gradcam/gradcam_EfficientNet_B0.png",
            "ResNet50":        "gradcam/gradcam_ResNet50.png",
            "DenseNet121":     "gradcam/gradcam_DenseNet121.png",
        },
    },
    "ISIC 2019": {
        "description": "9-class dermoscopic classification — AK, BCC, BKL, DF, MEL, NV, SCC, UNK, VASC.",
        "task": "multiclass",
        "num_classes": 9,
        "classes": ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "UNK", "VASC"],
        "metrics": {
            "EfficientNet-B0": {"Accuracy": 54.24, "Precision": 55.55, "Recall": 53.47, "F1": 51.13, "ROC-AUC": None},
            "ResNet50":        {"Accuracy": 58.47, "Precision": 65.08, "Recall": 62.96, "F1": 61.31, "ROC-AUC": None},
            "DenseNet121":     {"Accuracy": 61.02, "Precision": 57.87, "Recall": 59.03, "F1": 56.06, "ROC-AUC": None},
        },
        "plot_files": {
            "Class Distribution": "class_distribution.png",
            "Sample Images":      "sample_images.png",
            "Training Curves":    "training_curves.png",
            "Confusion Matrices": "confusion_matrices.png",
            "Model Comparison":   "model_comparison.png",
        },
        "gradcam_files": {},
    },
    "ISIC 2020": {
        "description": "Binary melanoma detection — Benign vs Malignant. 33,126 images, severe 98:2 class imbalance.",
        "task": "binary",
        "num_classes": 2,
        "classes": ["Benign", "Malignant"],
        "metrics": {
            "EfficientNet-B0": {"Accuracy": 92.85, "Precision": 97.07, "Recall": 92.85, "F1": 94.70, "ROC-AUC": 0.8833},
            "ResNet50":        {"Accuracy": 83.11, "Precision": 97.45, "Recall": 83.11, "F1": 89.04, "ROC-AUC": 0.8865},
            "DenseNet121":     {"Accuracy": 83.83, "Precision": 97.53, "Recall": 83.83, "F1": 89.49, "ROC-AUC": 0.8811},
        },
        "plot_files": {
            "Class Distribution":        "class_distribution.png",
            "Sample Images":             "sample_images.png",
            "ROC Curves":                "roc_curves.png",
            "Model Comparison":          "model_comparison.png",
            "EfficientNet-B0 Confusion": "confusion_efficientnet.png",
            "ResNet50 Confusion":        "confusion_resnet50.png",
            "DenseNet121 Confusion":     "confusion_densenet121.png",
        },
        "gradcam_files": {
            "All Models": "gradcam.png",
        },
    },
    "MILK10K": {
        "description": "11-class multi-centre dataset: AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC.",
        "task": "multiclass",
        "num_classes": 11,
        "classes": ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"],
        "metrics": {
            "EfficientNet-B0": {"Accuracy": 55.30, "Precision": 62.29, "Recall": 55.30, "F1": 57.55, "ROC-AUC": 0.8635},
            "ResNet50":        {"Accuracy": 52.39, "Precision": 62.02, "Recall": 52.39, "F1": 55.49, "ROC-AUC": 0.8380},
            "DenseNet121":     {"Accuracy": 60.50, "Precision": 64.25, "Recall": 60.50, "F1": 61.68, "ROC-AUC": 0.8709},
        },
        "plot_files": {
            "Class Distribution":        "class_distribution.png",
            "Sample Images":             "sample_images.png",
            "ROC Curves":                "roc_curves.png",
            "Model Comparison":          "model_comparison.png",
            "EfficientNet-B0 Confusion": "confusion_efficientnet.png",
            "ResNet50 Confusion":        "confusion_resnet50.png",
            "DenseNet121 Confusion":     "confusion_densenet121.png",
        },
        "gradcam_files": {
            "EfficientNet-B0": "gradcam_EfficientNet_B0.png",
            "ResNet50":        "gradcam_ResNet50.png",
            "DenseNet121":     "gradcam_DenseNet121.png",
        },
    },
}

# ── Preprocessing (ImageNet standard) ─────────────────────────────────────────
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

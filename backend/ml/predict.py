from .model import SkinLesionModel
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"

CLASSES = [
    "melanoma",
    "melanocytic_nevus",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "actinic_keratosis",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
    "seborrheic_keratosis",
    "lentigo_maligna",
    "lichen_planus",
    "psoriasis",
    "tinea",
    "urticaria",
]

# Force CPU for Render free tier
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Lazy loading
model = None

def get_model():
    global model

    if model is None:
        model = SkinLesionModel()

        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device)
        )

        model.to(device)
        model.eval()

    return model


@torch.no_grad()
def predict_image(image_path):

    model = get_model()

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    outputs = model(image)

    probs = torch.softmax(outputs, dim=1)[0]

    confidence, predicted = torch.max(probs, 0)

    top3 = torch.topk(probs, 3)

    return {
        "prediction": CLASSES[predicted.item()],
        "confidence": float(confidence.item()),

        "top3": [
            {
                "class": CLASSES[idx],
                "probability": round(float(prob) * 100, 2)
            }
            for prob, idx in zip(top3.values, top3.indices)
        ]
    }
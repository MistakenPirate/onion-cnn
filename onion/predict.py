import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from onion.model import CLASS_NAMES, OnionNet

MEAN = [0.7896, 0.6630, 0.6340]
STD = [0.2228, 0.3200, 0.3320]

_transform = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def load_model(weights_path: str = "model_weights.pth", device: torch.device | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OnionNet(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def predict_image(image: Image.Image, model: OnionNet, device: torch.device) -> dict:
    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    return {
        "prediction": CLASS_NAMES[pred_idx.item()],
        "confidence": round(confidence.item() * 100, 2),
    }

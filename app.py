"""
Kuluçkalık Yumurta Döllenme Durumu Sınıflandırma API'si
--------------------------------------------------------
EfficientNetB0 tabanlı, candling görüntülerinden fertile / infertile / dead
sınıflandırması yapan modeli local olarak servis eden FastAPI uygulaması.

Çalıştırma:
    pip install -r requirements.txt
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

Sonra tarayıcıda: http://localhost:8000
"""

import io
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !!! ÖNEMLİ !!!
# Bu sıralama, eğitim sırasında kullanılan sınıf sırasıyla BİREBİR aynı olmalı.
# PyTorch'ta ImageFolder, klasörleri alfabetik sıraya göre etiketler:
#   dead -> 0, fertile -> 1, infertile -> 2
# Eğer eğitimde farklı bir sıra kullandıysan (örn. class_to_idx.json'dan kontrol et),
# aşağıdaki listeyi ona göre düzelt.
CLASS_NAMES = ["fertile", "infertile", "dead"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def build_model() -> nn.Module:
    """Eğitimdeki ile ayni mimari: EfficientNetB0, son katman Linear(3)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    return model


def load_model() -> nn.Module:
    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Bazı eğitim script'leri sadece state_dict, bazıları
    # {"state_dict": ..., "epoch": ..., ...} gibi bir sözlük kaydeder. İkisini de destekle.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


app = FastAPI(title="Yumurta Döllenme Durumu Sınıflandırma API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model: nn.Module | None = None


@app.on_event("startup")
def startup_event():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model dosyası bulunamadı: {MODEL_PATH}\n"
            "Eğittiğin .pth dosyasını bu klasöre 'model.pth' adıyla kopyala."
        )
    model = load_model()
    print(f"[OK] Model yüklendi. Cihaz: {DEVICE}")


@app.get("/", response_class=HTMLResponse)
def root():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya türü: {file.content_type}. Sadece JPEG/PNG kabul edilir.",
        )

    raw = await file.read()
    try:
        # RGBA (PNG) -> RGB dönüşümü; eğitimdeki ön işleme adımıyla tutarlı.
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı, dosya bozuk olabilir.")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    all_probs = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(len(CLASS_NAMES))}
    predicted_idx = int(torch.argmax(probs).item())
    predicted_class = CLASS_NAMES[predicted_idx]

    return {
        "predicted_class": predicted_class,
        "confidence": all_probs[predicted_class],
        "all_probabilities": all_probs,
    }


# Statik dosyaları (varsa ek CSS/JS/görsel) sunmak için
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

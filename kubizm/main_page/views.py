from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os
import base64
from io import BytesIO


# ========== ОПРЕДЕЛЕНИЕ МОДЕЛИ ==========
class HybridModel(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridModel, self).__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ========================================

MODEL_PATH = os.path.join(settings.BASE_DIR, 'static_dev', 'python', 'hybrid_model.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = HybridModel(num_classes=3)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("✅ Модель загружена!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def Index(request):
    result = None
    preview_image = None

    if request.method == 'POST':
        if request.FILES.get('photo'):
            try:
                image_file = request.FILES.get('photo')

                # Сохраняем фото для предпросмотра (в base64)
                image = Image.open(image_file).convert('RGB')
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                preview_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Обрабатываем фото моделью
                if model is None:
                    result = {'error': 'Модель не загружена'}
                else:
                    input_tensor = transform(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, pred_idx = torch.max(probabilities, 1)

                        classes = {0: 'synthetic_cubism', 1: 'analytical_cubism', 2: 'other'}
                        pred_class = classes.get(pred_idx.item(), 'unknown')
                        confidence_value = float(confidence.item() * 100)

                        params = {
                            'geometry': round(float(probabilities[0][0].item()) * 100, 1),
                            'fragmentation': round(float(probabilities[0][1].item()) * 100, 1),
                            'palette': round(float(probabilities[0][2].item()) * 100, 1)
                        }

                    result = {
                        'class': pred_class.replace('_', ' ').title(),
                        'confidence': round(confidence_value, 2),
                        'params': params
                    }
            except Exception as e:
                result = {'error': str(e)}

    return render(request, 'main_page/index.html', {
        'result': result,
        'preview_image': preview_image
    })


def o_we(request):
    return render(request, 'main_page/o_we.html')
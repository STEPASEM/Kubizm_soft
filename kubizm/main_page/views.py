from django.shortcuts import render
from django.conf import settings
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os
import base64
from io import BytesIO
import tempfile
import sys
import warnings

warnings.filterwarnings("ignore")

# ========== ИМПОРТ features.py ==========
FEATURES_PATH = os.path.join(settings.BASE_DIR, 'static_dev', 'python')
if FEATURES_PATH not in sys.path:
    sys.path.insert(0, FEATURES_PATH)

from features import extract_features
# =========================================

# ========== ИМПОРТ МОДЕЛИ ARTWORK ==========
from .models import Artwork

# ===========================================

# ========== ПУТИ К ФАЙЛАМ ==========
MODEL_PATH = os.path.join(settings.BASE_DIR, 'static_dev', 'python', 'hybrid_model.pth')


# ===================================

# ========== ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ МОДЕЛИ ==========
class HybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super(HybridModel, self).__init__()
        # Backbone - ResNet18
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        # Классификатор: 512 (изображение) + 3 (признаки) = 515
        self.classifier = nn.Sequential(
            nn.Linear(515, 128),  # 515 -> 128
            nn.ReLU(),
            nn.Linear(128, num_classes)  # 128 -> 4
        )

    def forward(self, image_features, handcrafted_features):
        # Объединяем признаки
        combined = torch.cat([image_features, handcrafted_features], dim=1)
        output = self.classifier(combined)
        return output


# ====================================================

# ========== ЗАГРУЗКА МОДЕЛИ ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

print(f"🔍 Ищем модель: {MODEL_PATH}")
print(f"📁 Файл существует: {os.path.exists(MODEL_PATH)}")

if os.path.exists(MODEL_PATH):
    try:
        # Создаём экземпляр модели
        model = HybridModel(num_classes=4)

        # Загружаем веса (state_dict)
        state_dict = torch.load(MODEL_PATH, map_location=device)

        # Адаптируем ключи (если нужно)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key == 'classifier.0.weight':
                # Если веса имеют размер [128, 515], а наша модель ждёт [128, 515] - ок
                if value.shape[1] == 515:
                    new_state_dict[key] = value
                else:
                    # Обрезаем или адаптируем
                    new_state_dict[key] = value[:, :515] if value.shape[1] > 515 else value
            elif key == 'classifier.0.bias':
                new_state_dict[key] = value[:128] if len(value) > 128 else value
            elif key == 'classifier.2.weight':  # если в сохранённой модели слой 2
                new_state_dict['classifier.2.weight'] = value
            elif key == 'classifier.2.bias':
                new_state_dict['classifier.2.bias'] = value
            elif key == 'classifier.3.weight':  # если в сохранённой модели слой 3
                new_state_dict['classifier.2.weight'] = value
            elif key == 'classifier.3.bias':
                new_state_dict['classifier.2.bias'] = value
            else:
                new_state_dict[key] = value

        # Загружаем веса
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        print("✅ Модель успешно загружена!")
        print(f"📊 Архитектура загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback

        traceback.print_exc()
# =================================================

# ========== ПРЕОБРАЗОВАНИЯ ДЛЯ ИЗОБРАЖЕНИЯ ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =====================================================


def Index(request):
    result = None
    preview_image = None
    examples = None

    if request.method == 'POST':
        if request.FILES.get('photo'):
            try:
                image_file = request.FILES.get('photo')
                image = Image.open(image_file).convert('RGB')

                # Сохраняем фото для предпросмотра
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                preview_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                if model is None:
                    result = {'error': 'Модель не загружена'}
                else:
                    # Создаём временный файл для features.py
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        image.save(tmp_file, format='JPEG')
                        tmp_path = tmp_file.name

                    try:
                        # ===== 1. ВЫЧИСЛЯЕМ ПАРАМЕТРЫ через features.py =====
                        features_dict = extract_features(tmp_path)

                        params = {
                            'geometry': round(features_dict["geometry"] * 100, 1),
                            'fragmentation': round(features_dict["fragmentation"] * 100, 1),
                            'palette': round(features_dict["palette"] * 100, 1)
                        }

                        # ===== 2. ПОДГОТАВЛИВАЕМ ДАННЫЕ ДЛЯ МОДЕЛИ =====
                        # Признаки из изображения через backbone
                        image_tensor = transform(image).unsqueeze(0).to(device)

                        # Получаем признаки из backbone
                        with torch.no_grad():
                            img_features = model.backbone(image_tensor)  # [1, 512]

                        # Ручные признаки
                        handcrafted = torch.tensor([
                            features_dict["geometry"],
                            features_dict["fragmentation"],
                            features_dict["palette"]
                        ], dtype=torch.float32).unsqueeze(0).to(device)  # [1, 3]

                        # ===== 3. ЗАПУСКАЕМ МОДЕЛЬ =====
                        with torch.no_grad():
                            outputs = model(img_features, handcrafted)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, pred_idx = torch.max(probabilities, 1)

                            # Сопоставление индексов со стилями
                            style_map = {
                                0: 'analytical_cubism',
                                1: 'cubism',
                                2: 'not_cubism',
                                3: 'synthetic_cubism'
                            }

                            classes = {
                                0: 'Аналитический кубизм',
                                1: 'Кубизм',
                                2: 'Не кубизм',
                                3: 'Синтетический кубизм'
                            }

                            pred_class = classes.get(pred_idx.item(), 'unknown')
                            style_key = style_map.get(pred_idx.item())
                            confidence_value = float(confidence.item() * 100)

                        result = {
                            'class': pred_class,
                            'confidence': round(confidence_value, 1),
                            'params': params,
                            'style_key': style_key
                        }

                        # ===== 4. ПОЛУЧАЕМ ПРИМЕРЫ ИЗ БД =====
                        if style_key and style_key != 'not_cubism':
                            examples = Artwork.objects.filter(style=style_key)[:6]

                    finally:
                        os.unlink(tmp_path)

            except Exception as e:
                result = {'error': str(e)}
                print(f"Ошибка обработки: {e}")
                import traceback
                traceback.print_exc()

    return render(request, 'main_page/index.html', {
        'result': result,
        'preview_image': preview_image,
        'examples': examples
    })


def o_we(request):
    return render(request, 'main_page/o_we.html')
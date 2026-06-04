# ============================================================
# FEATURES.PY
#
# Вычисление признаков кубизма:
# 1. Геометризация формы
# 2. Фрагментация изображения
# 3. Нейтральность цветовой палитры
# ============================================================

import cv2
import numpy as np


# ============================================================
# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# ============================================================

def load_image(image_path):
    """
    Загружает изображение в формате OpenCV
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Не удалось загрузить изображение: {image_path}"
        )

    return image


# ============================================================
# ГЕОМЕТРИЗАЦИЯ
# ============================================================

def geometric_score(image_path):
    """
    Оценивает преобладание геометрических форм.

    Идея:
    - ищем контуры;
    - аппроксимируем их многоугольниками;
    - считаем долю контуров,
      имеющих >= 3 вершин.

    Результат:
    0.0 - почти нет геометрических фигур
    1.0 - преобладают многоугольники
    """

    image = load_image(image_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return 0.0

    polygon_count = 0

    for contour in contours:

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        if len(approx) >= 3:
            polygon_count += 1

    score = polygon_count / len(contours)

    return float(np.clip(score, 0, 1))


# ============================================================
# ФРАГМЕНТАЦИЯ
# ============================================================

def fragmentation_score(image_path):
    """
    Оценка фрагментации изображения.

    Идея:
    - используем Canny;
    - считаем плотность границ.

    Аналитический кубизм обычно
    имеет высокую плотность контуров.

    Результат:
    0.0 - гладкое изображение
    1.0 - высокая фрагментация
    """

    image = load_image(image_path)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_pixels = np.sum(edges > 0)

    total_pixels = edges.shape[0] * edges.shape[1]

    density = edge_pixels / total_pixels

    score = min(density * 10, 1.0)

    return float(score)


# ============================================================
# НЕЙТРАЛЬНОСТЬ ПАЛИТРЫ
# ============================================================

def palette_score(image_path):
    """
    Анализ цветовой палитры.

    Используем HSV.

    Низкая насыщенность (нейтральная):
        аналитический кубизм

    Высокая насыщенность (яркая):
        синтетический кубизм

    Результат:
    0.0 - нейтральная палитра (мало цвета)
    1.0 - яркая палитра (много цвета)
    """

    image = load_image(image_path)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]

    mean_saturation = np.mean(saturation)

    normalized = mean_saturation / 255.0

    # Убираем 1.0 - , теперь яркая палитра даёт высокий score
    score = normalized

    return float(np.clip(score, 0, 1))


# ============================================================
# ИТОГОВЫЙ ВЕКТОР ПРИЗНАКОВ
# ============================================================

def extract_features(image_path):
    """
    Возвращает все признаки разом.

    Пример:
    {
        "geometry": 0.81,
        "fragmentation": 0.72,
        "palette": 0.89
    }
    """

    return {
        "geometry": round(
            geometric_score(image_path),
            3
        ),

        "fragmentation": round(
            fragmentation_score(image_path),
            3
        ),


        "palette": round(
            palette_score(image_path),
            3
        )
    }
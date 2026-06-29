import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json


# -----------------------------
# DEVICE
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# CONFIG
# -----------------------------

IMG_SIZE = (224, 224)

CLASS_NAMES = ["1", "2", "5", "10"]
CLASS_VALUES = [1, 2, 5, 10]


# -----------------------------
# MODEL (тот же что в training)
# -----------------------------

model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    4
)

model.load_state_dict(torch.load("coin_model_fold5.pth", map_location=device))

model = model.to(device)
model.eval()


# -----------------------------
# TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------------
# DETECT COINS (HoughCircles)
# -----------------------------

def detect_coins(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=300,
        param1=150,
        param2=70,
        minRadius=150,
        maxRadius=400
    )

    if circles is None:
        return []

    circles = np.uint16(np.around(circles))

    return circles[0]


# -----------------------------
# CROP COIN
# -----------------------------

def crop_coin(image, x, y, r, margin=15):
    """
    Вырезает монету, накладывает круглую маску и
    возвращает изображение только с монетой.
    """

    # Координаты квадрата
    x1 = max(0, x - r - margin)
    y1 = max(0, y - r - margin)

    x2 = min(image.shape[1], x + r + margin)
    y2 = min(image.shape[0], y + r + margin)

    # Вырезаем область
    crop = image[y1:y2, x1:x2]

    # Размер вырезанной области
    h, w = crop.shape[:2]

    # Создаем черную маску
    mask = np.zeros((h, w), dtype=np.uint8)

    # Центр монеты относительно вырезанного изображения
    center = (x - x1, y - y1)

    # Радиус маски
    radius = r

    # Рисуем белый круг
    cv2.circle(mask, center, radius, 255, -1)

    # Оставляем только монету
    result = cv2.bitwise_and(crop, crop, mask=mask)

    return result

# -----------------------------
# CLASSIFY COIN
# -----------------------------

def predict_coin(coin_img):

    img = cv2.resize(coin_img, IMG_SIZE)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        pred = torch.argmax(outputs, dim=1).item()

    return CLASS_VALUES[pred]


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def process_image(image_path, visualize=True):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    circles = detect_coins(image)

    coins_result = []
    total_sum = 0

    vis = image.copy()

    for (x, y, r) in circles:

        coin_crop = crop_coin(image, x, y, r)

        try:
            value = predict_coin(coin_crop)
        except:
            value = 0

        coins_result.append(value)
        total_sum += value

        # ---------------- visualization ----------------
        if visualize:
            cv2.circle(vis, (x, y), r, (0, 255, 0), 3)
            cv2.putText(
                vis,
                str(value),
                (x - 20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    # ---------------- JSON OUTPUT ----------------

    result_json = {
        "total_sum": float(total_sum),
        "coins": [
            {"value": 1, "count": coins_result.count(1)},
            {"value": 2, "count": coins_result.count(2)},
            {"value": 5, "count": coins_result.count(5)},
            {"value": 10, "count": coins_result.count(10)}
        ]
    }

    print(json.dumps(result_json, indent=4, ensure_ascii=False))

    if visualize:
        cv2.imwrite("result.jpg", vis)

        print(json.dumps(result_json, indent=4, ensure_ascii=False))
        print("Изображение с результатом сохранено в result.jpg")
    return result_json


# -----------------------------
# RUN EXAMPLE
# -----------------------------

if __name__ == "__main__":

    image_path = "D:/practice 2nd year/Project/raw/scenes/IMG_3900.JPG"

    process_image(image_path)
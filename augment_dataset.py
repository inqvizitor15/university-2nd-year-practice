import os
import cv2
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------

INPUT_DIR = "dataset/all"
OUTPUT_DIR = "dataset/augmented"

CLASSES = ["1", "2", "5", "10"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# FUNCTIONS
# -----------------------------

def rotate(img, angle):

    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )


def brightness(img, alpha):

    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def contrast(img, beta):

    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def scale(img, s):

    h, w = img.shape[:2]

    resized = cv2.resize(
        img,
        None,
        fx=s,
        fy=s
    )

    canvas = np.zeros_like(img)

    rh, rw = resized.shape[:2]

    if s >= 1:

        x = (rw - w) // 2
        y = (rh - h) // 2

        return resized[y:y+h, x:x+w]

    else:

        x = (w-rw)//2
        y = (h-rh)//2

        canvas[y:y+rh, x:x+rw] = resized

        return canvas


def translate(img, dx, dy):

    h, w = img.shape[:2]

    M = np.float32([[1,0,dx],[0,1,dy]])

    return cv2.warpAffine(
        img,
        M,
        (w,h),
        borderMode=cv2.BORDER_REFLECT
    )


def blur(img,k):

    return cv2.GaussianBlur(img,(k,k),0)


def noise(img,std):

    n=np.random.normal(0,std,img.shape)

    out=img.astype(np.float32)+n

    return np.clip(out,0,255).astype(np.uint8)


def jpeg(img,quality):

    _,enc=cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY),quality]
    )

    return cv2.imdecode(enc,1)


def perspective(img):

    h,w=img.shape[:2]

    d=8

    src=np.float32([
        [0,0],
        [w,0],
        [0,h],
        [w,h]
    ])

    dst=np.float32([
        [d,d],
        [w-d,0],
        [0,h],
        [w,h-d]
    ])

    M=cv2.getPerspectiveTransform(src,dst)

    return cv2.warpPerspective(
        img,
        M,
        (w,h),
        borderMode=cv2.BORDER_REFLECT
    )

# -----------------------------
# AUGMENT
# -----------------------------

total = 0

for cls in CLASSES:

    input_folder = os.path.join(INPUT_DIR, cls)

    output_folder = os.path.join(OUTPUT_DIR, cls)

    os.makedirs(output_folder, exist_ok=True)

    files = sorted(os.listdir(input_folder))

    for idx, file in enumerate(files, start=1):

        if not file.lower().endswith((".jpg",".jpeg",".png")):
            continue

        img = cv2.imread(os.path.join(input_folder,file))

        if img is None:
            continue

        prefix = f"{idx:03d}"

        # ---------- original ----------

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_original.jpg"),
            img
        )

        # ---------- rotations ----------

        for angle in [30,60,90,120,150,180,240,300]:

            cv2.imwrite(
                os.path.join(output_folder,f"{prefix}_rot{angle}.jpg"),
                rotate(img,angle)
            )

        # ---------- brightness ----------

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_bright.jpg"),
            brightness(img,1.2)
        )

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_dark.jpg"),
            brightness(img,0.8)
        )

        # ---------- contrast ----------

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_contrast_plus.jpg"),
            contrast(img,20)
        )

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_contrast_minus.jpg"),
            contrast(img,-20)
        )

        # ---------- scale ----------

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_scale_up.jpg"),
            scale(img,1.08)
        )

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_scale_down.jpg"),
            scale(img,0.92)
        )

        # ---------- translate ----------

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_shift1.jpg"),
            translate(img,8,6)
        )

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_shift2.jpg"),
            translate(img,-8,-6)
        )

        # ---------- blur ----------

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_blur3.jpg"),
            blur(img,3)
        )

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_blur5.jpg"),
            blur(img,5)
        )

        # ---------- noise ----------

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_noise10.jpg"),
            noise(img,10)
        )

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_noise20.jpg"),
            noise(img,20)
        )

        # ---------- jpeg ----------

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_jpeg80.jpg"),
            jpeg(img,80)
        )

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_jpeg60.jpg"),
            jpeg(img,60)
        )

        # ---------- perspective ----------

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_perspective1.jpg"),
            perspective(img)
        )

        cv2.imwrite(
            os.path.join(output_folder,f"{prefix}_perspective2.jpg"),
            perspective(perspective(img))
        )

        total += 23

print("="*50)
print("Dataset successfully augmented.")
print("Total images:", total)
print("="*50)
import numpy as np
from PIL import Image, ImageFilter


CANVAS_SIZE = 280


def preprocess_pixels(pixels):
    if pixels.ndim != 2:
        raise ValueError("Expected a grayscale image.")

    foreground_y, foreground_x = np.where(pixels > 40)
    if len(foreground_y) < 1:
        return None

    cropped = pixels[
        foreground_y.min():foreground_y.max() + 1,
        foreground_x.min():foreground_x.max() + 1,
    ]
    height, width = cropped.shape
    side = max(1, int(max(height, width) * 1.4))
    square = np.zeros((side, side), dtype=np.uint8)
    top = (side - height) // 2
    left = (side - width) // 2
    square[top:top + height, left:left + width] = cropped

    small = Image.fromarray(square).resize((8, 8), Image.Resampling.LANCZOS)
    features = np.asarray(small, dtype=float)
    maximum = features.max()
    if maximum == 0:
        return None
    return (features / maximum * 16).reshape(-1)


def preprocess_training_digit(digit):
    pixels = np.rint(digit * (255 / 16)).clip(0, 255).astype(np.uint8)
    canvas = Image.fromarray(pixels).resize(
        (CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.BILINEAR
    )
    return preprocess_pixels(np.asarray(canvas))


def augment_training_digit(digit, rng, stroke_mode):
    pixels = np.rint(digit * (255 / 16)).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(pixels).resize(
        (CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.BILINEAR
    )
    image = image.rotate(
        float(rng.uniform(-10, 10)),
        resample=Image.Resampling.BICUBIC,
        translate=(
            int(rng.integers(-14, 15)),
            int(rng.integers(-14, 15)),
        ),
        fillcolor=0,
    )

    scale = float(rng.uniform(0.88, 1.12))
    if scale < 1:
        size = max(1, int(CANVAS_SIZE * scale))
        smaller = image.resize((size, size), Image.Resampling.BICUBIC)
        scaled = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        offset = (CANVAS_SIZE - size) // 2
        scaled.paste(smaller, (offset, offset))
        image = scaled
    else:
        size = max(1, int(CANVAS_SIZE / scale))
        offset = (CANVAS_SIZE - size) // 2
        image = image.crop(
            (offset, offset, offset + size, offset + size)
        ).resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.BICUBIC)

    if stroke_mode == "thick":
        kernel_size = int(rng.choice([23, 31, 39, 47]))
        image = image.filter(ImageFilter.MaxFilter(kernel_size))
    elif stroke_mode == "thin":
        kernel_size = int(rng.choice([3, 5]))
        image = image.filter(ImageFilter.MinFilter(kernel_size))

    return preprocess_pixels(np.asarray(image))

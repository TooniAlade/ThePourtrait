from p5 import *
from PIL import Image
import numpy as np
import os

# Config
WINDOW_WIDTH, WINDOW_HEIGHT = 512, 512
IMG_PATH = r"C:\Users\brand\OneDrive\Desktop\hackathon25\penguin_silhouette.png"

# Gradient stops
TOP_COLOR = Color(255, 120, 120)
MID_COLOR = Color(120, 200, 255)
BOTTOM_COLOR = Color(120, 255, 160)

img_np = None


def setup():
    global img_np
    size(WINDOW_WIDTH, WINDOW_HEIGHT)
    title("Penguin Pour - p5")
    no_stroke()
    # Load image via Pillow for fast pixel access
    if not os.path.exists(IMG_PATH):
        print(f"ERROR: Image not found at {IMG_PATH}")
        no_loop()
        return
    im = Image.open(IMG_PATH).convert("RGB")
    im = im.resize((WINDOW_WIDTH, WINDOW_HEIGHT), Image.LANCZOS)
    img_np = np.array(im)


def tri_gradient(t):
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        u = t / 0.5
        return lerp_color(TOP_COLOR, MID_COLOR, u)
    else:
        u = (t - 0.5) / 0.5
        return lerp_color(MID_COLOR, BOTTOM_COLOR, u)


def draw():
    background(250)
    if img_np is None:
        return

    # Create gradient underpainting masked by silhouette
    load_pixels()
    w, h = WINDOW_WIDTH, WINDOW_HEIGHT
    for y in range(h):
        t = y / (h - 1)
        col = tri_gradient(t)
        for x in range(w):
            r, g, b = img_np[y, x]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if lum < 80:  # inside silhouette
                pixels[y*w + x] = col
    update_pixels()

    # Add soft noise/texture using translucent ellipses
    random_seed(7)
    for i in range(1500):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        r, g, b = img_np[y, x]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum >= 80:
            continue
        t = y / (h - 1)
        base = tri_gradient(t)
        # jitter color
        jitter = lambda v: max(0, min(255, int(v + np.random.randint(-25, 26))))
        c = Color(jitter(base.red), jitter(base.green), jitter(base.blue), 120)
        fill(c)
        d = np.random.randint(3, 8)
        ellipse((x, y), d, d)

    no_loop()


if __name__ == "__main__":
    run() 

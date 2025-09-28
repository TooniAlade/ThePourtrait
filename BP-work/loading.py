import argparse
import json
import os
import random
import math
from typing import List, Tuple

import numpy as np
from PIL import Image
import pygame
from noise import pnoise2
import cv2 


WIDTH, HEIGHT = 800, 600
FPS = 60

DEFAULT_INPUT = "BP-work/silhouettes/rengoku.png"
DEFAULT_COLORS = "colors.json"
DEFAULT_OUTPUT = "marble_output.png"

BLOB_COLORS: List[Tuple[float, float, float]] = []
BLOB_WEIGHTS: List[float] = []


def parse_colors(colors_path: str):
    fallback_colors = [(230/255.0, 70/255.0, 70/255.0),
                       (60/255.0, 120/255.0, 230/255.0),
                       (60/255.0, 200/255.0, 120/255.0)]
    fallback_weights = [0.34, 0.33, 0.33]
    if not colors_path or not os.path.exists(colors_path):
        return fallback_colors, fallback_weights
    try:
        with open(colors_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return fallback_colors, fallback_weights

    colors, weights = [], []
    def hex_to_rgb01(h: str):
        h = h.strip().lstrip("#")
        return (int(h[0:2],16)/255.0, int(h[2:4],16)/255.0, int(h[4:6],16)/255.0)

    try:
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str):
                    colors.append(hex_to_rgb01(k)); weights.append(float(v))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "color" in item and "percent" in item:
                    col, pct = item["color"], item["percent"]
                    if isinstance(col, str):
                        colors.append(hex_to_rgb01(col)); weights.append(float(pct))
                    elif isinstance(col, (list, tuple)) and len(col) >= 3:
                        colors.append((col[0]/255.0, col[1]/255.0, col[2]/255.0)); weights.append(float(pct))
    except Exception:
        return fallback_colors, fallback_weights

    if not colors or not weights:
        return fallback_colors, fallback_weights
    w = np.array(weights, dtype=np.float64)
    w[w < 0] = 0
    s = float(w.sum())
    if s <= 0: return fallback_colors, fallback_weights
    return colors, (w / s).tolist()

# ----------------- SIMPLE MARBLING -----------------
def render_marble(input_path, colors_path, output_path, seed=0):
    img = Image.open(input_path).convert("L")
    W, H = img.size
    mask = np.array(img, dtype=np.float32) / 255.0
    inside = mask < 0.5

    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx+0.5)/W

    colors, weights = parse_colors(colors_path)
    colors_arr = np.array(colors, dtype=np.float32)

    t = u % 1.0
    cum = np.cumsum(weights)
    idx = np.digitize(t, cum)
    col = colors_arr[np.clip(idx, 0, len(colors_arr)-1)]

    out = np.ones((H,W,3), dtype=np.float32)
    out[inside] = col[inside]
    Image.fromarray(np.clip(out*255.0, 0, 255).astype(np.uint8)).save(output_path)


def pick_color(only_white=False):
    if only_white:
        return (255, 255, 255)
    r = random.random()
    cum = 0.0
    for color, weight in zip(BLOB_COLORS, BLOB_WEIGHTS):
        cum += weight
        if r <= cum:
            return (int(color[0]*255), int(color[1]*255), int(color[2]*255))
    c = BLOB_COLORS[-1]
    return (int(c[0]*255), int(c[1]*255), int(c[2]*255))

def spiral_wave_wrap(surface, t, alpha=255):
    mixed = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for y in range(HEIGHT):
        offset = int(40 * math.sin((y / 60.0) + t))
        row = surface.subsurface((0, y, WIDTH, 1)).copy()
        row.set_alpha(alpha)
        mixed.blit(row, (offset, y))
        mixed.blit(row, (offset - WIDTH, y))
        mixed.blit(row, (offset + WIDTH, y))
    return mixed

def run_animation(reveal_path: str):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Paint Pour → Spiral → Blob Reveal")
    clock = pygame.time.Clock()

    full_img = pygame.image.load(reveal_path).convert()
    full_img = pygame.transform.scale(full_img, (WIDTH, HEIGHT))

    paint_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    paint_blobs = []
    reveal_blobs = []

    phase = 1
    spiral_timer = 0.0
    frame = 0
    running = True
    only_white = False

    def add_blob(color, is_reveal=False):
        blob = {
            "x": WIDTH//2 + random.randint(-60, 60) if not is_reveal else random.randint(100, WIDTH-100),
            "y": HEIGHT//2 + random.randint(-60, 60) if not is_reveal else random.randint(100, HEIGHT-100),
            "r": 5,
            "max_r": random.randint(250, 500) if not is_reveal else random.randint(120, 250),
            "growth": random.uniform(6.0, 9.0),  # much faster growth
            "spikiness": random.uniform(25, 45) if not is_reveal else random.uniform(30, 60),
            "jitter": random.randint(3, 6),
            "color": color,
            "life": 0
        }
        if is_reveal:
            reveal_blobs.append(blob)
        else:
            paint_blobs.append(blob)

    def update_paint_blobs():
        for b in paint_blobs:
            if b["r"] < b["max_r"]:
                b["r"] += b["growth"]
            b["life"] += 1
            s = pygame.Surface((int(b["r"]*2+40), int(b["r"]*2+40)), pygame.SRCALPHA)
            points = []
            for angle in range(0, 360, b["jitter"]):
                rad = math.radians(angle)
                noise = pnoise2(math.cos(rad)*0.7 + b["life"]*0.01,
                                math.sin(rad)*0.7 + b["life"]*0.01,
                                base=int(b["life"]*0.2))
                noisy_r = b["r"] + noise * b["spikiness"]
                x = int(b["r"] + math.cos(rad) * noisy_r + 20)
                y = int(b["r"] + math.sin(rad) * noisy_r + 20)
                points.append((x, y))
            pygame.draw.polygon(s, b["color"] + (255,), points)
            paint_surface.blit(s, (int(b["x"]-b["r"]-20), int(b["y"]-b["r"]-20)))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Phase 1: colored blobs fill
        if phase == 1:
            if frame % 2 == 0:
                add_blob(pick_color())
            update_paint_blobs()
            screen.blit(paint_surface, (0, 0))
            if frame > 90:   # 1.5 seconds
                only_white = True
                phase = 2

        # --- Phase 2: only white blobs + spiral 
        elif phase == 2:
            if frame % 2 == 0:
                add_blob(pick_color(only_white=True))
            update_paint_blobs()
            spiral_timer += 0.15   # faster spiral motion
            mixed = spiral_wave_wrap(paint_surface, spiral_timer, 255)
            screen.blit(mixed, (0, 0))
            if spiral_timer > 2:  # 2 seconds max
                phase = 3

        # --- Phase 3: reveal with splotchy blobs 
       # --- inside Phase 3 ---
        elif phase == 3:
            if frame % 2 == 0:
                add_blob(pick_color(only_white=True))
            update_paint_blobs()
            spiral_timer += 0.15
            mixed_paint = spiral_wave_wrap(paint_surface, spiral_timer, 255)
            screen.blit(mixed_paint, (0, 0))

            if frame % 3 == 0 and len(reveal_blobs) < 40:  # allow more blobs
                add_blob((255, 255, 255), is_reveal=True)

            arr_paint = pygame.surfarray.array3d(mixed_paint).swapaxes(0, 1)
            arr_img   = pygame.surfarray.array3d(full_img).swapaxes(0, 1)

            mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

            for rb in reveal_blobs:
                if rb["r"] < rb["max_r"]:
                    rb["r"] += rb["growth"]
                rb["life"] += 1

                points = []
                for angle in range(0, 360, rb.get("jitter", 5)):
                    rad = math.radians(angle)
                    noise = pnoise2(math.cos(rad)*0.7 + rb["life"]*0.01,
                                    math.sin(rad)*0.7 + rb["life"]*0.01,
                                    base=int(rb["life"]*0.2))
                    noisy_r = rb["r"] + noise * rb.get("spikiness", 40)
                    x = int(rb["x"] + math.cos(rad) * noisy_r)
                    y = int(rb["y"] + math.sin(rad) * noisy_r)
                    points.append((x, y))

                if len(points) > 2:
                    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)

            mask = mask.astype(bool)
            arr_paint[mask] = arr_img[mask]

            surface_out = pygame.surfarray.make_surface(arr_paint.swapaxes(0, 1))
            screen.blit(surface_out, (0, 0))

            # --- completion check ---
            revealed_fraction = mask.mean()
            if revealed_fraction > 0.95 or frame > 400: 
                phase = 4


        # --- Phase 4: final static image ---
        elif phase == 4:
            screen.blit(full_img, (0, 0))

        pygame.display.flip()
        clock.tick(FPS)
        frame += 1

    pygame.quit()

# ----------------- MAIN -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--colors", default=DEFAULT_COLORS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    global BLOB_COLORS, BLOB_WEIGHTS
    BLOB_COLORS, BLOB_WEIGHTS = parse_colors(args.colors)

    render_marble(args.input, args.colors, args.output, seed=0)
    run_animation(args.output)

if __name__ == "__main__":
    main()

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

# Defaults
DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "wolfimage.png")
DEFAULT_COLORS = os.path.join(os.path.dirname(__file__), "colors.json")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "marble_output.png")


def parse_colors(colors_path: str) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """
    Parse colors and ratios from a JSON file. Accepts a few flexible formats:
    - [{"color": "#RRGGBB", "percent": 30}, ...]
    - [{"color": [r,g,b], "percent": 30}, ...]
    - {"#RRGGBB": 30, "#0000FF": 50, "#00FF00": 20}
    - [[r,g,b, percent], ...]

    Returns (colors_rgb_0_1, weights_norm) where weights sum to 1.
    If file missing or invalid, returns a fallback palette.
    """
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

    colors = []
    weights = []

    def hex_to_rgb01(h: str):
        h = h.strip()
        if h.startswith("#"):
            h = h[1:]
        if len(h) == 3:  # #RGB
            r = int(h[0]*2, 16)
            g = int(h[1]*2, 16)
            b = int(h[2]*2, 16)
        else:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
        return (r/255.0, g/255.0, b/255.0)

    try:
        if isinstance(data, dict):
            for k, v in data.items():
                # k is color, v is percent
                if isinstance(k, str):
                    c = hex_to_rgb01(k)
                elif isinstance(k, (list, tuple)) and len(k) >= 3:
                    c = (float(k[0])/255.0, float(k[1])/255.0, float(k[2])/255.0)
                else:
                    continue
                colors.append(c)
                weights.append(float(v))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "color" in item and "percent" in item:
                        col = item["color"]
                        pct = item["percent"]
                        if isinstance(col, str):
                            c = hex_to_rgb01(col)
                        elif isinstance(col, (list, tuple)) and len(col) >= 3:
                            c = (float(col[0])/255.0, float(col[1])/255.0, float(col[2])/255.0)
                        else:
                            continue
                        colors.append(c)
                        weights.append(float(pct))
                elif isinstance(item, (list, tuple)) and len(item) >= 4:
                    c = (float(item[0])/255.0, float(item[1])/255.0, float(item[2])/255.0)
                    colors.append(c)
                    weights.append(float(item[3]))
    except Exception:
        return fallback_colors, fallback_weights

    # normalize
    if not colors or not weights:
        return fallback_colors, fallback_weights
    w = np.array(weights, dtype=np.float64)
    w[w < 0] = 0
    s = float(w.sum())
    if s <= 0:
        return fallback_colors, fallback_weights
    w = (w / s).tolist()
    return colors, w


def build_bands(weights: List[float]) -> np.ndarray:
    """
    Convert weights into cumulative bounds in [0,1], last element = 1.
    Use sqrt on cumulative so band areas are proportional in circular warps,
    but since we stripe along x, we keep linear cumulative for now.
    """
    cum = np.cumsum(np.array(weights, dtype=np.float64))
    cum[-1] = 1.0
    return cum


def smoothstep(edge0, edge1, x):
    # Elementwise smoothstep with safe denominator for arrays
    denom = np.maximum(1e-8, (edge1 - edge0))
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def domain_warp(u, v, seed=0, strength=0.1, octaves=3):
    """Generate a displacement field (du,dv) using summed sines/cosines for marbling."""
    rng = np.random.RandomState(seed)
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    amp = strength
    for o in range(octaves):
        # random orientation and frequency
        a1 = rng.uniform(-1.0, 1.0)
        b1 = rng.uniform(-1.0, 1.0)
        a2 = rng.uniform(-1.0, 1.0)
        b2 = rng.uniform(-1.0, 1.0)
        f = rng.uniform(1.5, 4.5) * (1.6 ** o)
        p1 = rng.uniform(0, 2*np.pi)
        p2 = rng.uniform(0, 2*np.pi)
        theta1 = 2*np.pi * (a1 * u + b1 * v) * f + p1
        theta2 = 2*np.pi * (a2 * u + b2 * v) * f + p2
        du += amp * (np.sin(theta1) + 0.5*np.cos(theta2))
        dv += amp * (np.cos(theta1) - 0.5*np.sin(theta2))
        amp *= 0.5
    return du, dv


def swirl_field(u, v, centers: List[Tuple[float,float]], strength=0.06, radius=0.35):
    """Add rotational displacement around given centers with Gaussian falloff."""
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    rsq = radius * radius
    for (cx, cy) in centers:
        rx = u - cx
        ry = v - cy
        r2 = rx*rx + ry*ry
        fall = np.exp(-r2 / max(1e-6, rsq))
        invr = 1.0 / np.sqrt(r2 + 1e-6)
        du += strength * (-ry) * invr * fall
        dv += strength * ( rx) * invr * fall
    return du, dv


def render_marble(input_path: str, colors_path: str, output_path: str, seed: int = 0,
                  edge: float = 0.015, warp_strength: float = 0.12, octaves: int = 3,
                  swirl_count: int = 2) -> None:
    # Load silhouette as mask
    img = Image.open(input_path).convert("L")
    W, H = img.size
    mask = np.array(img, dtype=np.float32) / 255.0
    inside = mask > 0.5

    # Coordinates in [0,1]
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx + 0.5) / float(W)
    v = (yy + 0.5) / float(H)

    # Load colors and weights
    colors, weights = parse_colors(colors_path)
    bounds = build_bands(weights)  # 1D cumulative
    colors_arr = np.array(colors, dtype=np.float32)  # (N,3)

    # Base stripe coordinate (with a slight vertical wobble)
    rng = np.random.RandomState(seed)
    wobble = 0.07 * np.sin(2*np.pi*(1.7*v + rng.uniform(0,1)))
    t = (u + wobble) % 1.0

    # Domain warp for marbling
    du1, dv1 = domain_warp(u, v, seed=seed, strength=warp_strength, octaves=octaves)
    # Big swirls
    # Choose swirl centers within the silhouette bbox
    ys, xs = np.where(inside)
    if ys.size > 0:
        y_min, y_max = ys.min()/float(H), ys.max()/float(H)
        x_min, x_max = xs.min()/float(W), xs.max()/float(W)
    else:
        y_min, y_max, x_min, x_max = 0.2, 0.8, 0.2, 0.8
    centers = [(rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)) for _ in range(max(0, swirl_count))]
    du2, dv2 = swirl_field(u, v, centers, strength=warp_strength*0.7, radius=0.35)

    u2 = (u + du1 + du2) % 1.0
    v2 = (v + dv1 + dv2) % 1.0
    t2 = (u2 + 0.0*v2) % 1.0  # stripes primarily along x, warped by displacement

    # Map t2 to color bands with soft edges
    # Ensure last bound = 1.0
    bounds = np.array(bounds, dtype=np.float32)
    bounds[-1] = 1.0

    # Find band index: first i where t2 <= bounds[i]
    cmp = t2[..., None] <= bounds[None, None, :]
    # Guarantee at least one True at the last slot
    cmp[..., -1] = True
    idx = np.argmax(cmp, axis=-1)  # (H,W)

    # Current and previous colors
    curr_col = colors_arr[idx]
    prev_idx = np.clip(idx - 1, 0, len(bounds)-1)
    prev_col = colors_arr[prev_idx]
    prev_bound = np.where(idx > 0, bounds[idx - 1], 0.0)

    # Smooth blend near boundary
    w = smoothstep(prev_bound, prev_bound + edge, t2)
    w = np.where(idx > 0, w, 1.0)
    col = prev_col * (1.0 - w)[..., None] + curr_col * w[..., None]

    # Composite over white outside mask
    out = np.ones((H, W, 3), dtype=np.float32)
    out[inside] = col[inside]

    # Save
    Image.fromarray(np.clip(out*255.0, 0, 255).astype(np.uint8)).save(output_path)


def main():
    p = argparse.ArgumentParser(description="Generate a marbled image inside a silhouette using color ratios.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Path to silhouette image (PNG with black silhouette on white)")
    p.add_argument("--colors", default=DEFAULT_COLORS, help="Path to colors.json with ratios")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save output PNG")
    p.add_argument("--seed", type=int, default=0, help="Random seed for marbling")
    p.add_argument("--edge", type=float, default=0.015, help="Soft edge width between bands (0..0.1)")
    p.add_argument("--strength", type=float, default=0.12, help="Warp strength (0..0.3)")
    p.add_argument("--octaves", type=int, default=3, help="Warp octaves (1..5)")
    p.add_argument("--swirls", type=int, default=2, help="Number of large swirl centers (0..5)")
    args = p.parse_args()

    render_marble(args.input, args.colors, args.output, seed=args.seed,
                  edge=args.edge, warp_strength=args.strength, octaves=args.octaves,
                  swirl_count=args.swirls)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

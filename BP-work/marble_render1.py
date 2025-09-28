import argparse
import json
import os
import random
import time
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
try:
    import pygame
    from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

# Defaults
def rgb01_to_hex(c):
    """(r,g,b) in 0..1 -> '#rrggbb'"""
    r = int(round(max(0.0, min(1.0, c[0])) * 255))
    g = int(round(max(0.0, min(1.0, c[1])) * 255))
    b = int(round(max(0.0, min(1.0, c[2])) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"

def hex_to_rgb01(h: str):
    """'#rrggbb' or 'rgb' -> (r,g,b) in 0..1"""
    h = h.strip().lstrip('#')
    if len(h) == 3:  # #RGB
        r = int(h[0]*2, 16); g = int(h[1]*2, 16); b = int(h[2]*2, 16)
    else:
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (r/255.0, g/255.0, b/255.0)

def get_random_silhouette():
    """Get a random silhouette from the silhouettes folder, fallback to wolfimage.png if none found."""
    silhouettes_dir = os.path.join(os.path.dirname(__file__), "silhouettes")
    fallback = os.path.join(os.path.dirname(__file__), "wolfimage.png")
    
    if not os.path.exists(silhouettes_dir):
        return fallback
    
    # Get all PNG files from silhouettes directory
    silhouette_files = [f for f in os.listdir(silhouettes_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not silhouette_files:
        return fallback
    
    # Select random silhouette
    selected_file = random.choice(silhouette_files)
    selected_path = os.path.join(silhouettes_dir, selected_file)
    
    try:
        print(f"🎲 Randomly selected silhouette: {selected_file}")
    except UnicodeEncodeError:
        print(f"[DICE] Randomly selected silhouette: {selected_file}")
    return selected_path

DEFAULT_INPUT = get_random_silhouette()
DEFAULT_COLORS = os.path.join(os.path.dirname(__file__), "colors.json")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "marble_output.png")


def parse_colors(colors_path: str) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """
    Parse colors from JSON file and randomly select 6 colors with standard percentages.
    Automatically picks 6 colors from the full palette and applies:
    25%, 20%, 20%, 15%, 10%, 10% distribution.
    
    Returns (colors_rgb_0_1, weights_norm) where weights sum to 1.
    If file missing or invalid, returns a fallback palette.
    """
    fallback_colors = [(230/255.0, 70/255.0, 70/255.0),
                       (60/255.0, 120/255.0, 230/255.0),
                       (60/255.0, 200/255.0, 120/255.0)]
    fallback_weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
    
    # Standard 6-color percentage breakdown
    standard_percentages = [25, 20, 20, 15, 10, 10]

    try:
        with open(colors_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return fallback_colors[:6], [w/100.0 for w in standard_percentages]

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
    # Parse all available colors from the JSON
    all_colors = []
    
    try:
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "color" in item:
                    col = item["color"]
                    if isinstance(col, str):
                        c = hex_to_rgb01(col)
                        all_colors.append(c)
                    elif isinstance(col, (list, tuple)) and len(col) >= 3:
                        c = (float(col[0])/255.0, float(col[1])/255.0, float(col[2])/255.0)
                        all_colors.append(c)
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    c = (float(item[0])/255.0, float(item[1])/255.0, float(item[2])/255.0)
                    all_colors.append(c)
    except Exception:
        return fallback_colors[:6], [w/100.0 for w in standard_percentages]

    # If we have colors, randomly select 6 and apply standard percentages
    if all_colors and len(all_colors) >= 6:
        selected_colors = random.sample(all_colors, 6)
        selected_weights = [w/100.0 for w in standard_percentages]
        
        # Show which colors were selected
        color_names = []
        for color in selected_colors:
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            color_names.append(hex_color)
        try:
            print(f"🎨 Selected colors: {', '.join(color_names)}")
            print(f"📊 Percentages: {standard_percentages}")
        except UnicodeEncodeError:
            print(f"[ART] Selected colors: {', '.join(color_names)}")
            print(f"[CHART] Percentages: {standard_percentages}")
        
        return selected_colors, selected_weights
    elif all_colors:
        # If fewer than 6 colors, use what we have with adjusted percentages
        selected_weights = [w/100.0 for w in standard_percentages[:len(all_colors)]]
        # Normalize to sum to 1
        total = sum(selected_weights)
        selected_weights = [w/total for w in selected_weights]
        return all_colors, selected_weights
    else:
        return fallback_colors[:6], [w/100.0 for w in standard_percentages]


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
                  edge: float = 0.018, warp_strength: float = 0.14, octaves: int = 4,
                  swirl_count: int = 2, comb_amp: float = 0.05, comb_freq: float = 10.0,
                  flow_steps: int = 2, palette_out: Optional[str] = None) -> None:
    # Load silhouette as mask
    img = Image.open(input_path).convert("L")
    W, H = img.size
    mask = np.array(img, dtype=np.float32) / 255.0
    # Invert: treat dark silhouette as inside, white background as outside
    inside = mask < 0.5

    # Coordinates in [0,1]
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx + 0.5) / float(W)
    v = (yy + 0.5) / float(H)

    # Load colors and weights
    colors, weights = parse_colors(colors_path)
    # write the palette we actually used
    if palette_out:
        with open(palette_out, "w", encoding="utf-8") as f:
            json.dump(
                {"colors": [rgb01_to_hex(c) for c in colors], "weights": weights},
                f, ensure_ascii=False
            )
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
    # Comb (rake) warp: introduce repeated streaks along v to increase marbling
    phase = np.random.RandomState(seed+77).uniform(0.0, 1.0)
    u2 = (u2 + comb_amp * np.sin(2*np.pi*(comb_freq * v + phase))) % 1.0
    # Iterative flow to elongate streaks (advect coordinates a couple of times)
    for i in range(max(0, int(flow_steps))):
        duf, dvf = domain_warp(u2, v2, seed=seed+200+i, strength=warp_strength*0.6, octaves=max(1, octaves-1))
        u2 = (u2 + 0.5 * duf) % 1.0
        v2 = (v2 + 0.5 * dvf) % 1.0
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


def compute_center(inside_mask: np.ndarray) -> Tuple[float, float]:
    """Return center (cx, cy) in pixel coordinates as the centroid of the inside mask.
    If mask is empty, return image center.
    """
    H, W = inside_mask.shape
    ys, xs = np.where(inside_mask)
    if ys.size == 0:
        return (W/2.0, H/2.0)
    cx = xs.mean()
    cy = ys.mean()
    return (float(cx), float(cy))


def animate_reveal(col: np.ndarray, inside: np.ndarray, output_path: str,
                   center: Optional[Tuple[float,float]] = None,
                   seconds: float = 6.0, fps: int = 30, reveal_edge_px: float = 10.0,
                   preview: bool = True, save_final: bool = True) -> None:
    """
    Animate a radial reveal from center over white background. If preview is True and pygame
    is available, display live; otherwise, save a few progress frames and the final image.
    col: (H,W,3) float32 in [0,1]
    inside: (H,W) boolean mask
    center: (cx,cy) in pixel coordinates; if None, use centroid
    """
    H, W, _ = col.shape
    if center is None:
        cx, cy = compute_center(inside)
    else:
        cx, cy = center

    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    # Maximum distance needed to cover all inside pixels
    max_dist = float(dist[inside].max()) if inside.any() else float(np.hypot(W, H))
    total_frames = max(1, int(seconds * fps))

    def frame_image(i: int) -> np.ndarray:
        t = i / max(1, total_frames - 1)
        r = t * max_dist
        edge = reveal_edge_px
        # inside weight = 1 inside radius, 0 outside, with soft edge
        w = 1.0 - smoothstep(r - edge, r + edge, dist)
        w = w * (inside.astype(np.float32))
        w3 = w[..., None]
        base = np.ones((H, W, 3), dtype=np.float32)
        return base * (1.0 - w3) + col * w3

    if preview and HAS_PYGAME:
        pygame.init()
        surf = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Marble Reveal Preview (Esc to exit)")
        clock = pygame.time.Clock()

        running = True
        # Animation phase
        for i in range(total_frames):
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False
            if not running:
                break
            frame = frame_image(i)
            frame8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            frame_surface = pygame.surfarray.make_surface(np.transpose(frame8, (1, 0, 2)))
            surf.blit(frame_surface, (0, 0))
            pygame.display.flip()
            clock.tick(fps)

        # Hold final image on screen until user exits
        if running:
            final = frame_image(total_frames - 1)
            final8 = np.clip(final * 255.0, 0, 255).astype(np.uint8)
            final_surface = pygame.surfarray.make_surface(np.transpose(final8, (1, 0, 2)))
            surf.blit(final_surface, (0, 0))
            pygame.display.flip()
            while running:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False
                    if event.type == KEYDOWN and event.key == K_ESCAPE:
                        running = False
                clock.tick(30)

        pygame.quit()

    # Always save final image
    if save_final:
        final = frame_image(total_frames - 1)
        Image.fromarray(np.clip(final*255.0, 0, 255).astype(np.uint8)).save(output_path)


def main():
    p = argparse.ArgumentParser(description="Generate a marbled image inside a silhouette using color ratios. Randomly selects from silhouettes folder by default.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Path to silhouette image (PNG with black silhouette on white). If not specified, randomly selects from silhouettes/ folder")
    p.add_argument("--random", action="store_true", help="Force random silhouette selection even if --input is specified")
    p.add_argument("--list-silhouettes", action="store_true", help="List available silhouettes and exit")
    p.add_argument("--colors", default=DEFAULT_COLORS, help="Path to colors.json with ratios")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save output PNG")
    p.add_argument("--seed", type=int, default=3, help="Random seed for marbling")
    p.add_argument("--edge", type=float, default=0.02, help="Soft edge width between bands (0..0.1)")
    p.add_argument("--strength", type=float, default=0.16, help="Warp strength (0..0.3)")
    p.add_argument("--octaves", type=int, default=4, help="Warp octaves (1..5)")
    p.add_argument("--swirls", type=int, default=2, help="Number of large swirl centers (0..5)")
    p.add_argument("--comb-amp", type=float, default=0.06, help="Comb (rake) warp amplitude (0..0.2)")
    p.add_argument("--comb-freq", type=float, default=9.0, help="Comb (rake) warp frequency (bands along vertical)")
    p.add_argument("--flow-steps", type=int, default=2, help="Iterative flow steps to elongate streaks (0..4)")
    # Reveal animation options - PREVIEW IS NOW ON BY DEFAULT
    p.add_argument("--no-preview", action="store_true", help="Disable the live center-out reveal animation")
    p.add_argument("--seconds", type=float, default=6.0, help="Duration of the reveal animation")
    p.add_argument("--fps", type=int, default=30, help="Frames per second for preview")
    p.add_argument("--center-x", type=float, default=None, help="Reveal center X in [0,1] (relative); default: mask centroid")
    p.add_argument("--center-y", type=float, default=None, help="Reveal center Y in [0,1] (relative); default: mask centroid")
    p.add_argument("--reveal-edge", type=float, default=12.0, help="Reveal edge softness in pixels")
    p.add_argument("--palette-out", default=None, help="Path to write the selected colors/weights JSON")

    args = p.parse_args()

    # Handle special options
    if args.list_silhouettes:
        silhouettes_dir = os.path.join(os.path.dirname(__file__), "silhouettes")
        if os.path.exists(silhouettes_dir):
            silhouette_files = [f for f in os.listdir(silhouettes_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if silhouette_files:
                try:
                    print("🎨 Available silhouettes:")
                except UnicodeEncodeError:
                    print("[ART] Available silhouettes:")
                for i, filename in enumerate(sorted(silhouette_files), 1):
                    print(f"  {i}. {filename}")
            else:
                try:
                    print("❌ No silhouettes found in silhouettes/ folder")
                except UnicodeEncodeError:
                    print("[X] No silhouettes found in silhouettes/ folder")
        else:
            try:
                print("❌ silhouettes/ folder not found")
            except UnicodeEncodeError:
                print("[X] silhouettes/ folder not found")
        return

    # Use random silhouette if --random flag is used
    if args.random:
        args.input = get_random_silhouette()

    # Set preview to True by default, disable only if --no-preview is used
    args.preview = not args.no_preview

    # First, generate the marbled image deterministically for the given seed/params
    render_marble(
        args.input, args.colors, args.output, seed=args.seed,
        edge=args.edge, warp_strength=args.strength, octaves=args.octaves,
        swirl_count=args.swirls, comb_amp=args.comb_amp, comb_freq=args.comb_freq,
        flow_steps=args.flow_steps, palette_out=args.palette_out
    )

    # If no preview requested, we're done
    if not getattr(args, "preview", False):
        print(f"Saved: {args.output}")
        return

    # If preview requested, compute the same image in-memory and animate reveal
    img = Image.open(args.output).convert("RGB")
    col = np.array(img, dtype=np.float32) / 255.0
    # Re-load mask and inside for reveal and center calculation
    mask_img = Image.open(args.input).convert("L")
    mask = np.array(mask_img, dtype=np.float32) / 255.0
    inside = mask < 0.5

    H, W, _ = col.shape
    if args.center_x is not None and args.center_y is not None:
        cx = float(np.clip(args.center_x, 0.0, 1.0)) * W
        cy = float(np.clip(args.center_y, 0.0, 1.0)) * H
        center = (cx, cy)
    else:
        center = None

    animate_reveal(col, inside, args.output,
                   center=center,
                   seconds=args.seconds,
                   fps=args.fps,
                   reveal_edge_px=args.reveal_edge,
                   preview=True,
                   save_final=True)
    print(f"Saved (final frame): {args.output}")


if __name__ == "__main__":
    main()

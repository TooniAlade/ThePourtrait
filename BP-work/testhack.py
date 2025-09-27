import pygame
import random
import os

pygame.init()

# === Paths ===
image_path = r"C:\Users\brand\OneDrive\Desktop\hackathon25\penguin_silhouette.png"

# === Config (tweak these) ===
WINDOW_WIDTH, WINDOW_HEIGHT = 512, 512
SUPERSAMPLE = 2  # 1=off, 2 or 3 for smoother results
DOT_COUNT = 6000  # number of dots to spray (at supersampled resolution)
DOT_RADIUS_RANGE = (2, 6)  # radius in pixels at supersampled scale
BRIGHTNESS_THRESHOLD = 80  # how dark a pixel must be to count as "inside" silhouette
TOP_COLOR = (255, 120, 120)  # gradient top color
MID_COLOR = (120, 200, 255)  # optional mid color (used in 3-stop gradient)
BOTTOM_COLOR = (120, 255, 160)  # gradient bottom color
SHOW_OUTLINE = True

# === Utilities ===
def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def lerp_color(c1, c2, t: float):
    t = clamp01(t)
    return (
        int(lerp(c1[0], c2[0], t)),
        int(lerp(c1[1], c2[1], t)),
        int(lerp(c1[2], c2[2], t)),
    )

def tri_gradient(t: float):
    """Three-stop vertical gradient using TOP -> MID -> BOTTOM."""
    t = clamp01(t)
    if t < 0.5:
        return lerp_color(TOP_COLOR, MID_COLOR, t / 0.5)
    else:
        return lerp_color(MID_COLOR, BOTTOM_COLOR, (t - 0.5) / 0.5)

def brightness(rgb) -> int:
    # perceptual-ish luminance
    r, g, b = rgb[0], rgb[1], rgb[2]
    return int(0.2126 * r + 0.7152 * g + 0.0722 * b)

def build_alpha_mask_from_brightness(surface: pygame.Surface, threshold: int) -> pygame.Surface:
    """Create an alpha mask Surface (RGBA) where inside=opaque, outside=transparent based on brightness threshold."""
    w, h = surface.get_width(), surface.get_height()
    mask_surf = pygame.Surface((w, h), flags=pygame.SRCALPHA)
    # Per-pixel loop; fine for ~512x512
    get_at = surface.get_at
    mask_set_at = mask_surf.set_at
    for y in range(h):
        for x in range(w):
            p = get_at((x, y))
            if brightness(p) < threshold:
                mask_set_at((x, y), (255, 255, 255, 255))
            else:
                mask_set_at((x, y), (255, 255, 255, 0))
    return mask_surf

def point_inside(surface: pygame.Surface, x: int, y: int, threshold: int) -> bool:
    p = surface.get_at((x, y))
    return brightness(p) < threshold

def draw_alpha_circle(target: pygame.Surface, color_rgba, center, radius: int):
    # Draw to a small SRCALPHA surface, then blit for proper soft edges under supersampling downsizing
    d = radius * 2
    blob = pygame.Surface((d, d), flags=pygame.SRCALPHA)
    pygame.draw.circle(blob, color_rgba, (radius, radius), radius)
    target.blit(blob, (center[0] - radius, center[1] - radius))

# === Load image ===
if not os.path.exists(image_path):
    print(f"ERROR: Image not found at {image_path}")
    pygame.quit()
    raise SystemExit(1)

src_img = pygame.image.load(image_path).convert()
src_w, src_h = src_img.get_width(), src_img.get_height()
print(f"Loaded penguin silhouette: {src_w} x {src_h}")

# === Window ===
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Penguin Pour (Smooth)")

# === Prepare working surfaces ===
hi_w, hi_h = WINDOW_WIDTH * SUPERSAMPLE, WINDOW_HEIGHT * SUPERSAMPLE
hires = pygame.Surface((hi_w, hi_h), flags=pygame.SRCALPHA)

# Fit source image into working size while preserving aspect ratio
scale = min(hi_w / src_w, hi_h / src_h)
dst_w, dst_h = int(src_w * scale), int(src_h * scale)
offset_x = (hi_w - dst_w) // 2
offset_y = (hi_h - dst_h) // 2

scaled = pygame.transform.smoothscale(src_img, (dst_w, dst_h))
mask_src = pygame.Surface((hi_w, hi_h))
mask_src.fill((255, 255, 255))
mask_src.blit(scaled, (offset_x, offset_y))

# Build silhouette alpha mask once and keep for compositing
silhouette_alpha = build_alpha_mask_from_brightness(mask_src, BRIGHTNESS_THRESHOLD)

# === Paint background gradient (underpainting) ===
gradient = pygame.Surface((hi_w, hi_h), flags=pygame.SRCALPHA)
for y in range(hi_h):
    t = y / max(1, (hi_h - 1))
    col = tri_gradient(t)
    pygame.draw.line(gradient, (*col, 255), (0, y), (hi_w, y))

# Mask the gradient so it only shows inside silhouette
gradient.blit(silhouette_alpha, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
hires.blit(gradient, (0, 0))

# === Spray semi-transparent colored dots for texture/interest ===
for i in range(DOT_COUNT):
    x = random.randint(0, hi_w - 1)
    y = random.randint(0, hi_h - 1)
    if not point_inside(mask_src, x, y, BRIGHTNESS_THRESHOLD):
        continue

    t = y / max(1, (hi_h - 1))
    base_col = tri_gradient(t)
    # Slight random tint + alpha for variation
    jitter = lambda v: max(0, min(255, v + random.randint(-25, 25)))
    col = (jitter(base_col[0]), jitter(base_col[1]), jitter(base_col[2]))
    alpha = random.randint(60, 130)
    radius = random.randint(*DOT_RADIUS_RANGE)
    draw_alpha_circle(hires, (*col, alpha), (x, y), radius)

# Optional: draw a subtle outline around the silhouette
if SHOW_OUTLINE:
    # Use a coarse edge by expanding/eroding the mask via offset sampling
    outline = pygame.Surface((hi_w, hi_h), flags=pygame.SRCALPHA)
    px = mask_src
    getp = px.get_at
    outline_col = (20, 20, 20, 100)
    for y in range(1, hi_h - 1):
        for x in range(1, hi_w - 1):
            c = brightness(getp((x, y))) < BRIGHTNESS_THRESHOLD
            if not c:
                # If any neighbor is inside, mark border
                inside_n = (
                    brightness(getp((x + 1, y))) < BRIGHTNESS_THRESHOLD or
                    brightness(getp((x - 1, y))) < BRIGHTNESS_THRESHOLD or
                    brightness(getp((x, y + 1))) < BRIGHTNESS_THRESHOLD or
                    brightness(getp((x, y - 1))) < BRIGHTNESS_THRESHOLD
                )
                if inside_n:
                    outline.set_at((x, y), outline_col)
    hires.blit(outline, (0, 0))

# === Downsample to window for built-in smoothing ===
final_img = pygame.transform.smoothscale(hires, (WINDOW_WIDTH, WINDOW_HEIGHT))

# === Display ===
screen.fill((250, 250, 250))
screen.blit(final_img, (0, 0))
pygame.display.flip()

# === Event Loop ===
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

pygame.quit()

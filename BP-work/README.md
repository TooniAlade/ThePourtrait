# ThePourtrait
HackGT12 Project with cup pouring of paint on a digital canvas.





# Penguin Pour - Smoother Aesthetics

This workspace contains three ways to render your penguin silhouette with color blending and nicer visuals:

1) Pygame (improved) – works out-of-the-box with your current file.
2) p5.py sketch – simple creative-coding style, anti-aliased drawing.
3) PyOpenGL – GPU-shaded gradient with anti-banding noise and alpha mask.

## Files
- `testhack.py`: Upgraded Pygame renderer using supersampling, gradient blending, and alpha-blended dots plus optional outline.
- `p5_sketch.py`: p5.py version that loads the silhouette, paints a smooth vertical gradient inside, and sprinkles translucent dots.
- `opengl_render.py`: Minimal PyOpenGL example that masks the silhouette on the GPU and renders a smooth tri-stop gradient with subtle noise.
- `penguin_silhouette.png`: Your source silhouette image.

## Python dependencies
Pick the path you want to try:

- Pygame only: `pygame`
- p5.py path: `p5`, `Pillow`, `numpy`
- PyOpenGL path: `pygame`, `PyOpenGL`, `Pillow`, `numpy`

### Install (Windows PowerShell)
```powershell
# It's OK if some are already installed
python -m pip install --upgrade pip
python -m pip install pygame
python -m pip install p5 Pillow numpy
python -m pip install PyOpenGL PyOpenGL_accelerate
```

If you use a virtual environment, activate it before installing.

## Run
- Improved Pygame (recommended starting point):
```powershell
python .\testhack.py
```

- p5.py version:
```powershell
python .\p5_sketch.py
```

- PyOpenGL version:
```powershell
python .\opengl_render.py
```

## Tweaks
Open `testhack.py` and adjust:
- `SUPERSAMPLE`: 1 (off), 2, or 3. Higher is smoother but slower.
- `DOT_COUNT`, `DOT_RADIUS_RANGE`: density and size of texture dots.
- `BRIGHTNESS_THRESHOLD`: how dark a pixel must be to count as inside the silhouette.
- `TOP_COLOR`, `MID_COLOR`, `BOTTOM_COLOR`: the gradient.
- `SHOW_OUTLINE`: toggles subtle border.

## Notes
- If performance is slow, lower `SUPERSAMPLE` or `DOT_COUNT`.
- If your silhouette is very small or very large, the Pygame version scales to fit the window via supersampling.
- The PyOpenGL example needs graphics drivers that support OpenGL 3.3+.

# Minimal PyOpenGL renderer for smooth gradient fill inside a silhouette
# Requires: pygame, PyOpenGL, numpy, pillow

import os
import sys
import numpy as np
from PIL import Image
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL import GL as gl

WINDOW_WIDTH, WINDOW_HEIGHT = 900, 900
IMG_PATH = r"C:\Users\brand\OneDrive\Desktop\hackathon25\wolfimage.png"
COLORS_JSON = r"C:\Users\brand\OneDrive\Desktop\hackathon25\colors.json"
BRIGHTNESS_THRESHOLD = 80  # lower = darker required to count as inside
FILL_SPEED = 0.25  # fraction per second (0..1/sec)
MAX_SWIRLS = 12  # number of recent swirl centers to drive marbling
MAX_COLORS = 8   # maximum colors in the target-style brush
ENABLE_WARP = False  # solid target brush; no marbling
ENABLE_WARP = False  # solid layers only; no marbling

# Fallback tri-gradient if colors.json missing
TOP_COLOR = (255/255.0, 120/255.0, 120/255.0)
MID_COLOR = (120/255.0, 200/255.0, 255/255.0)
BOTTOM_COLOR = (120/255.0, 255/255.0, 160/255.0)

VERT_SRC = """
#version 330 core
layout(location=0) in vec2 in_pos;
layout(location=1) in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 out_col;

uniform sampler2D u_mask;
uniform sampler2D u_canvas; // accumulated poured color (premultiplied RGBA)

float rand(vec2 co){
    // simple hash for subtle noise
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453);
}

void main(){
    // Flip V when sampling texture to match image origin (top-left in Pillow) to OpenGL's bottom-left UVs
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec4 m = texture(u_mask, uv);
    if(m.a < 0.5) discard;

    // Present accumulated canvas color (premultiplied RGBA)
    vec4 col = texture(u_canvas, v_uv);
    // Optional tiny noise for anti-banding on low alpha edges
    float n = rand(v_uv * 1024.0) * 0.02 - 0.01;
    col.rgb = clamp(col.rgb + n, 0.0, 1.0);
    // Composite over white background to avoid see-through look
    // since col is premultiplied: out = col.rgb + bg*(1 - col.a)
    vec3 bg = vec3(1.0);
    vec3 out_rgb = col.rgb + bg * (1.0 - col.a);
    out_col = vec4(out_rgb, 1.0);
}
"""

# UPDATED: Splat shader now takes NON-premultiplied u_color and premultiplies inside shader
SPLAT_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 out_col;

uniform sampler2D u_mask;
uniform vec2 u_center; // in UV (0..1)
uniform float u_radius; // in UV units relative to min(width,height)
uniform vec4 u_color; // NON-premultiplied RGBA (rgb, alpha)
uniform int u_use_mask; // 1 to respect silhouette mask, 0 to ignore (for overlay)

void main(){
    // distance in UV
    vec2 p = v_uv;
    float d = distance(p, u_center);
    // Soft circular falloff: 1 inside, 0 outside
    float fall = 1.0 - smoothstep(u_radius*0.6, u_radius, d);
    if(u_use_mask == 1){
        vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
        vec4 m = texture(u_mask, uv);
        fall *= m.a;
    }
    if(fall <= 0.001) discard;
    // Convert to premultiplied output:
    float out_a = u_color.a * fall;
    vec3 out_rgb = u_color.rgb * out_a; // premultiply here
    out_col = vec4(out_rgb, out_a);
}
"""

SPLAT_TARGET_FRAG = """
#version 330 core
#define MAX_COLORS 8
in vec2 v_uv;
out vec4 out_col;

uniform sampler2D u_mask;
uniform vec2  u_center;   // UV
uniform float u_radius;   // UV
uniform float u_alpha;    // 0..1
uniform int   u_use_mask; // 1 clip by silhouette
uniform int   u_count;    // number of color bands (<= MAX_COLORS)
uniform vec3  u_colors[MAX_COLORS];
uniform float u_bounds[MAX_COLORS]; // cumulative normalized radii in (0,1], last=1
uniform float u_edge;     // soften outer edge width in normalized radius

void main(){
    vec2 dp = v_uv - u_center;
    float r = length(dp);
    if(r > u_radius) discard;
    float rn = r / u_radius;

    if(u_use_mask == 1){
        vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
        float m = texture(u_mask, uv).a;
        if(m < 0.5) discard;
    }

    int idx = 0;
    for(int i=0;i<u_count;i++){
        if(rn <= u_bounds[i]){ idx = i; break; }
    }
    vec3 col = u_colors[idx];

    // Soft outer edge only
    float fade = 1.0 - smoothstep(1.0 - u_edge, 1.0, rn);
    float a = u_alpha * fade;
    out_col = vec4(col * a, a);
}
"""

CUP_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 out_col;

uniform vec2 u_cup_center; // UV
uniform vec2 u_cup_size;   // UV width,height
uniform float u_cup_rot;   // radians
uniform vec4 u_cup_color;  // premultiplied RGBA

// 2D rotation
mat2 rot(float a){
    float c = cos(a), s = sin(a);
    return mat2(c,-s,s,c);
}

// Signed distance to rounded rectangle
float sdRoundRect(vec2 p, vec2 b, float r){
    vec2 q = abs(p) - b + vec2(r);
    return length(max(q, 0.0)) - r + min(max(q.x, q.y), 0.0);
}

// Signed distance to ellipse centered at origin with radii a,b
float sdEllipse(vec2 p, vec2 r){
    return (length(p / r) - 1.0) * min(r.x, r.y);
}

void main(){
    // Convert to local cup space
    vec2 p = (v_uv - u_cup_center);
    // scale to UV units so size works uniformly
    vec2 cup_half = 0.5 * u_cup_size;
    p = rot(-u_cup_rot) * (p / cup_half);

    // Body: rounded rectangle (width=2,height=2 in this local space), rounded bottom
    float body = sdRoundRect(p + vec2(0.0, 0.05), vec2(0.9, 0.9), 0.2);

    // Rim: thin ellipse at top
    float rim = sdEllipse(p - vec2(0.0, 0.95), vec2(0.8, 0.2));

    // Handle: ring on right side
    vec2 hp = p - vec2(1.05, 0.1);
    float outer = length(hp) - 0.55;
    float inner = length(hp) - 0.35;
    float handle = max(outer, -inner); // ring region where outer<=0 and inner>=0

    // Combine shapes: inside if body<=0 or handle<=0 or near rim
    float d = min(min(body, handle), abs(rim) - 0.02);
    float a = smoothstep(0.03, -0.01, d); // soft edge
    if(a <= 0.001) discard;

    // Simple shading: slight vertical gradient
    float shade = clamp(0.9 + 0.1 * (1.0 - (p.y+1.0)*0.5), 0.8, 1.0);
    vec3 col = u_cup_color.rgb * shade;
    out_col = vec4(col * a, u_cup_color.a * a);
}
"""

WARP_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 out_col;

uniform sampler2D u_src;   // source canvas (premultiplied)
uniform sampler2D u_mask;  // silhouette mask
uniform float u_time;
uniform int   u_count;     // number of active swirls
uniform vec2  u_centers[NSWIRLS]; // UV centers
uniform float u_radii[NSWIRLS];   // UV radii
uniform float u_strengths[NSWIRLS];
uniform vec2  u_flow_dir;  // base flow direction (unit-ish)
uniform float u_flow;      // base flow amount
uniform float u_mix;       // mix warped vs base sample to conserve mass

void main(){
    vec2 uv = v_uv;
    // Only warp inside the mask
    vec2 muv = vec2(uv.x, 1.0 - uv.y);
    float m = texture(u_mask, muv).a;
    if(m < 0.5){
        out_col = vec4(0.0);
        return;
    }

    vec2 disp = vec2(0.0);
    // subtle base flow to stretch ribbons along scan direction
    disp += u_flow_dir * (u_flow * 0.003);

    // Sum of local swirls
    for(int i=0;i<u_count;i++){
        vec2 c = u_centers[i];
        float R = u_radii[i];
        float S = u_strengths[i];
        vec2 d = uv - c;
        float r = length(d);
        if(r < R && R > 1e-4){
            float t = 1.0 - (r / R);
            // normalized radial and tangential
            vec2 n = d / max(r, 1e-4);
            vec2 tang = vec2(-n.y, n.x);
            // swirl more near the center, taper outward
            float amp = S * (t*t);
            // tangential swirl
            disp += tang * (amp * 0.035);
            // slight radial push to create cells and borders
            disp += n * (amp * 0.015);
        }
    }

    // sample with backward mapping (avoid gaps)
    vec2 suv = clamp(uv - disp, 0.0, 1.0);
    vec4 colWarp = texture(u_src, suv);
    vec4 colBase = texture(u_src, uv);
    // Mix warped with base to avoid draining; preserve maximum coverage in alpha
    float mixv = clamp(u_mix, 0.0, 1.0);
    vec3 rgb = mix(colBase.rgb, colWarp.rgb, mixv); // premultiplied rgb mix
    float a   = max(colBase.a, colWarp.a);
    out_col = vec4(rgb, a) * m; // premultiplied preserved and masked
}
"""
WARP_FRAG = WARP_FRAG.replace("NSWIRLS", str(MAX_SWIRLS))


def compile_shader(src, shader_type):
    sid = gl.glCreateShader(shader_type)
    gl.glShaderSource(sid, src)
    gl.glCompileShader(sid)
    if gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        err = gl.glGetShaderInfoLog(sid).decode()
        raise RuntimeError(f"Shader compile error: {err}")
    return sid


def build_program(vs_src, fs_src):
    vs = compile_shader(vs_src, gl.GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, gl.GL_FRAGMENT_SHADER)
    pid = gl.glCreateProgram()
    gl.glAttachShader(pid, vs)
    gl.glAttachShader(pid, fs)
    gl.glLinkProgram(pid)
    if gl.glGetProgramiv(pid, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        err = gl.glGetProgramInfoLog(pid).decode()
        raise RuntimeError(f"Program link error: {err}")
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return pid


def make_fullscreen_quad():
    # positions (x,y) and uvs (u,v)
    verts = np.array([
        #   x,   y,   u,  v
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
         1.0,  1.0, 1.0, 1.0,
        -1.0,  1.0, 0.0, 1.0,
    ], dtype=np.float32)
    idx = np.array([0,1,2, 0,2,3], dtype=np.uint32)

    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    ebo = gl.glGenBuffers(1)

    gl.glBindVertexArray(vao)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)

    stride = 4 * 4  # 4 floats per vertex * 4 bytes
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(8))

    gl.glBindVertexArray(0)
    return vao, vbo, ebo


def load_mask_texture(path, w, h, threshold):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    im = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
    arr = np.array(im)
    # luminance
    lum = (0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2])
    inside = (lum < threshold).astype(np.uint8) * 255
    # We'll upload as RGBA, alpha=inside
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[...,0:3] = 255
    rgba[...,3] = inside

    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tex


def create_canvas_fbo(w, h):
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)

    fbo = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex, 0)
    status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("Framebuffer incomplete")
    # clear to transparent
    gl.glViewport(0, 0, w, h)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return fbo, tex


def load_color_percentages(json_path):
    import json
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        items = data.get("colors", [])
        if not items:
            raise ValueError("colors.json missing 'colors' array")
        amounts = [max(0.0, float(c.get("amount", 0))) for c in items]
        total = sum(amounts) or 1.0
        perc = [a/total for a in amounts]
        rgbs = [tuple((np.clip(c.get("rgb", [255,255,255]), 0, 255))) for c in items]
    else:
        # fallback to gradient stops equally distributed
        perc = [1/3, 1/3, 1/3]
        rgbs = [
            (int(TOP_COLOR[0]*255), int(TOP_COLOR[1]*255), int(TOP_COLOR[2]*255)),
            (int(MID_COLOR[0]*255), int(MID_COLOR[1]*255), int(MID_COLOR[2]*255)),
            (int(BOTTOM_COLOR[0]*255), int(BOTTOM_COLOR[1]*255), int(BOTTOM_COLOR[2]*255)),
        ]
    # build cumulative distribution
    cum = []
    acc = 0.0
    for p in perc:
        cum.append((acc, acc + p))
        acc += p
    return cum, rgbs


def main():
    pygame.init()
    # Try to enable multisampling (anti-aliasing)
    try:
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    except Exception:
        pass
    pygame.display.set_caption("Penguin Pour - PyOpenGL")
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), OPENGL | DOUBLEBUF)

    gl.glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    gl.glEnable(gl.GL_BLEND)
    # We'll switch blending modes per pass
    if gl.glGetString(gl.GL_VERSION):
        try:
            gl.glEnable(gl.GL_MULTISAMPLE)
        except Exception:
            pass

    prog_present = build_program(VERT_SRC, FRAG_SRC)
    prog_splat = build_program(VERT_SRC, SPLAT_FRAG)
    prog_target= build_program(VERT_SRC, SPLAT_TARGET_FRAG)
    prog_cup   = build_program(VERT_SRC, CUP_FRAG)  # retained but not used
    prog_warp  = build_program(VERT_SRC, WARP_FRAG)
    vao, vbo, ebo = make_fullscreen_quad()

    mask_tex = load_mask_texture(IMG_PATH, WINDOW_WIDTH, WINDOW_HEIGHT, BRIGHTNESS_THRESHOLD)
    # Create two FBOs/textures for ping-pong warping
    fbo_a, canvas_tex_a = create_canvas_fbo(WINDOW_WIDTH, WINDOW_HEIGHT)
    fbo_b, canvas_tex_b = create_canvas_fbo(WINDOW_WIDTH, WINDOW_HEIGHT)
    # Current canvas (source for present and destination for splats)
    cur_fbo, cur_tex = fbo_a, canvas_tex_a
    alt_fbo, alt_tex = fbo_b, canvas_tex_b

    # Present uniforms
    gl.glUseProgram(prog_present)
    loc_mask_present = gl.glGetUniformLocation(prog_present, "u_mask")
    loc_canvas_present = gl.glGetUniformLocation(prog_present, "u_canvas")
    gl.glUniform1i(loc_mask_present, 0)
    gl.glUniform1i(loc_canvas_present, 1)

    # Splat uniforms
    gl.glUseProgram(prog_splat)
    loc_mask_splat = gl.glGetUniformLocation(prog_splat, "u_mask")
    loc_center = gl.glGetUniformLocation(prog_splat, "u_center")
    loc_radius = gl.glGetUniformLocation(prog_splat, "u_radius")
    loc_color = gl.glGetUniformLocation(prog_splat, "u_color")
    loc_use_mask = gl.glGetUniformLocation(prog_splat, "u_use_mask")
    gl.glUniform1i(loc_mask_splat, 0)

    # Target brush uniforms
    gl.glUseProgram(prog_target)
    loc_t_mask   = gl.glGetUniformLocation(prog_target, "u_mask")
    loc_t_center = gl.glGetUniformLocation(prog_target, "u_center")
    loc_t_radius = gl.glGetUniformLocation(prog_target, "u_radius")
    loc_t_alpha  = gl.glGetUniformLocation(prog_target, "u_alpha")
    loc_t_use    = gl.glGetUniformLocation(prog_target, "u_use_mask")
    loc_t_count  = gl.glGetUniformLocation(prog_target, "u_count")
    loc_t_colors = gl.glGetUniformLocation(prog_target, "u_colors[0]")
    loc_t_bounds = gl.glGetUniformLocation(prog_target, "u_bounds[0]")
    loc_t_edge   = gl.glGetUniformLocation(prog_target, "u_edge")
    gl.glUniform1i(loc_t_mask, 0)

    # Cup uniforms
    gl.glUseProgram(prog_cup)
    loc_cup_center = gl.glGetUniformLocation(prog_cup, "u_cup_center")
    loc_cup_size   = gl.glGetUniformLocation(prog_cup, "u_cup_size")
    loc_cup_rot    = gl.glGetUniformLocation(prog_cup, "u_cup_rot")
    loc_cup_color  = gl.glGetUniformLocation(prog_cup, "u_cup_color")

    # Load color percentages
    cum_ranges, colors_rgb = load_color_percentages(COLORS_JSON)
    def current_color_for(t):
        # t in [0,1]
        for (a,b), rgb in zip(cum_ranges, colors_rgb):
            if t >= a and t < b:
                return tuple([c/255.0 for c in rgb])
        return tuple([c/255.0 for c in colors_rgb[-1]])

    # Build target brush arrays from percentage amounts
    percs = [max(0.0, b - a) for (a,b) in cum_ranges]
    total_p = sum(percs) or 1.0
    percs = [p/total_p for p in percs]
    cum = 0.0
    bounds = []
    for p in percs:
        cum += p
        bounds.append(np.sqrt(max(0.0, min(1.0, cum))))
    n_colors = min(len(bounds), MAX_COLORS)
    brush_bounds = np.zeros((MAX_COLORS,), dtype=np.float32)
    for i in range(n_colors):
        brush_bounds[i] = float(bounds[i])
    brush_colors = np.zeros((MAX_COLORS,3), dtype=np.float32)
    for i in range(n_colors):
        rgb = colors_rgb[i]
        brush_colors[i,0] = rgb[0]/255.0
        brush_colors[i,1] = rgb[1]/255.0
        brush_colors[i,2] = rgb[2]/255.0

    clock = pygame.time.Clock()
    time_s = 0.0
    stripe_phase = 0.0  # unused when solid layers
    # Recent swirl centers buffer (store uvx, uvy, spawnTime)
    swirls = []
    last_swirl_add_t = 0.0
    # Serpentine scan state
    lanes = 18
    left = WINDOW_WIDTH * 0.1
    right = WINDOW_WIDTH * 0.9
    top = WINDOW_HEIGHT * 0.15
    bottom = WINDOW_HEIGHT * 0.9
    lane_h = max(1.0, (bottom - top) / max(1, (lanes - 1)))
    lane_idx = 0
    dir_sign = 1  # 1 = left->right, -1 = right->left
    x_pos = left
    scan_speed = 180.0  # pixels/sec (slower for more coverage)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                lane_idx = 0
                dir_sign = 1
                x_pos = left

        # advance scanning position based on frame time
        float_dt = clock.get_time() / 1000.0
        time_s += float_dt
    # stripe_phase unused in solid mode
        x_pos += dir_sign * scan_speed * float_dt
        # clamp and handle lane switching
        if dir_sign > 0 and x_pos >= right:
            x_pos = right
            dir_sign = -1
            lane_idx += 1
        elif dir_sign < 0 and x_pos <= left:
            x_pos = left
            dir_sign = 1
            lane_idx += 1
        if lane_idx >= lanes:
            # loop back to top for continuous coverage
            lane_idx = 0
        # compute cup and impact positions
        lane_y = top + lane_idx * lane_h
        impact_x = x_pos
        impact_y = lane_y
        cx = x_pos
        cy = max(top - 40.0, lane_y - 60.0)  # cup sits above impact

        # keep a trail of swirl centers only if warp is enabled
        if ENABLE_WARP:
            if (time_s - last_swirl_add_t) > 0.12:
                last_swirl_add_t = time_s
                swirls.append((impact_x/float(WINDOW_WIDTH), impact_y/float(WINDOW_HEIGHT), time_s))
                if len(swirls) > MAX_SWIRLS:
                    swirls = swirls[-MAX_SWIRLS:]

        # Choose current stream color based on cumulative percentages
        # derive total coverage fraction from serpentine path
        span = (right - left)
        if span <= 0: span = 1.0
        if dir_sign > 0:
            frac_x = (x_pos - left) / span
        else:
            frac_x = (right - x_pos) / span
        poured_t = np.clip((lane_idx + frac_x) / float(lanes), 0.0, 1.0)
        alpha_stream = 1.0  # solid coating
        # single solid color at a time, chosen from cumulative percentages
        cur_rgb = current_color_for(poured_t)

        # Render splats to current canvas FBO (respecting mask)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, cur_fbo)
        gl.glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        gl.glDisable(gl.GL_BLEND)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, mask_tex)
        gl.glUseProgram(prog_target)
        gl.glUniform1i(loc_t_use, 1)
        gl.glUniform1i(loc_t_count, int(n_colors))
        if loc_t_colors != -1:
            gl.glUniform3fv(loc_t_colors, MAX_COLORS, brush_colors)
        if loc_t_bounds != -1:
            gl.glUniform1fv(loc_t_bounds, MAX_COLORS, brush_bounds)
        gl.glUniform1f(loc_t_edge, 0.08)

        gl.glBindVertexArray(vao)
        # draw a single target brush at the current impact point
        u = impact_x / WINDOW_WIDTH
        v = impact_y / WINDOW_HEIGHT
        gl.glUniform2f(loc_t_center, u, v)
        gl.glUniform1f(loc_t_radius, 0.05)
        gl.glUniform1f(loc_t_alpha, 1.0)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)

        # no splash/bleed in solid-target mode
        gl.glBindVertexArray(0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        if ENABLE_WARP:
            # Warp step (disabled)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, alt_fbo)
            gl.glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
            gl.glDisable(gl.GL_BLEND)
            gl.glUseProgram(prog_warp)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, mask_tex)
            gl.glActiveTexture(gl.GL_TEXTURE2)
            gl.glBindTexture(gl.GL_TEXTURE_2D, cur_tex)
            loc_warp_mask = gl.glGetUniformLocation(prog_warp, "u_mask")
            loc_warp_src  = gl.glGetUniformLocation(prog_warp, "u_src")
            gl.glUniform1i(loc_warp_mask, 0)
            gl.glUniform1i(loc_warp_src, 2)
            gl.glBindVertexArray(vao)
            gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            # swap for next frame
            cur_fbo, alt_fbo = alt_fbo, cur_fbo
            cur_tex, alt_tex = alt_tex, cur_tex

        # Present to screen
        gl.glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        gl.glDisable(gl.GL_BLEND)
        gl.glClearColor(250/255.0, 250/255.0, 250/255.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Bind mask and canvas textures
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, mask_tex)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, cur_tex)

        gl.glUseProgram(prog_present)
        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        pygame.display.flip()
        clock.tick(60)

    # Cleanup
    gl.glDeleteTextures([mask_tex, cur_tex, alt_tex])
    gl.glDeleteProgram(prog_present)
    gl.glDeleteProgram(prog_splat)
    gl.glDeleteProgram(prog_target)
    gl.glDeleteProgram(prog_cup)
    gl.glDeleteBuffers(1, [vbo])
    gl.glDeleteBuffers(1, [ebo])
    gl.glDeleteVertexArrays(1, [vao])
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        pygame.quit()
        sys.exit(1)

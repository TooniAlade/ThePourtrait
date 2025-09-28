# 🎨 ThePourtrait Gallery

Human Interaction based Marble art generator with Arduino-sourced color palettes, random or chosen silhouettes, and email delivery of high‑res images.

This document is the single source of truth. Other READMEs now point here.

## 🚀 Quick Start

Windows PowerShell

1) First-time setup
- Clone repo, then run:
	- `./setup_for_teammates.bat`
- Start the app:
	- `cd BP-work`
	- `./start_with_email.bat`
- Open: http://localhost:5000

2) Daily use
- `cd BP-work` → `./start_with_email.bat`

What the setup script does
- Creates a Python venv
- Installs requirements
- Prepares launch script

## 🌟 Features

- Random or chosen silhouettes (modal selector)
- Deterministic palette from Arduino (first 6 colors; weights 25/20/20/15/10/10)
- Reset button captures Arduino colors and clears canvas
- Human-friendly color names in UI and email
- Email with inline preview image and high‑res attachment
- Download button for high‑res PNG

## 📁 Key Files

- `BP-work/gallery_server1.py` — Flask server (active)
- `BP-work/marble_render1.py` — Renderer; writes sidecar palette JSON including silhouette
- `BP-work/marble_gallery1.html` — Front-end UI
- `BP-work/arduino_to_marble.py` — Reads six colors from Arduino to `colors.json`
- `BP-work/colors.json` — Latest captured colors
- `BP-work/generated_art/` — Output images + `.palette.json`
- `BP-work/silhouettes/` — Silhouette PNG/JPGs
- `BP-work/requirements.txt` — Python deps
- `BP-work/start_with_email.bat` — Launcher

## ⚙️ Configuration

Environment variables
- `EMAIL_USER` — Gmail address (defaults to `pourtrait12@gmail.com`)
- `EMAIL_PASS` — Gmail App Password (no spaces). Required to send email.
- `TEST_MODE` — if `true`, may alter behavior; production should set to `false`.
- `ARD_PORT` — Optional fixed serial port (e.g., `COM3`). Auto-detected if not set.
- `ARD_BAUD` — Baud rate (default `115200`).
- `ARD_COLOR_TIMEOUT` — Seconds to wait for 6 Arduino colors (`0`=no timeout). UI reset uses 120s.

Email notes
- Uses Gmail SMTP with STARTTLS (587) and SSL fallback (465).
- Email includes HTML body with color names, inline image preview, and a separate attachment.

## 🧪 Using the App

Generate art
- Click “Generate New Marble Art”
- Optionally open modal to choose a silhouette (otherwise random)

Reset (capture Arduino colors)
- Click “Reset” to read six colors via serial and overwrite `colors.json`
- Canvas clears; then click Generate to use new palette

Email / Download
- Enter name and email; click “Send to Email”
- Or click “Download Art” for a high‑res PNG

## 🎨 Color Naming

- Front-end and backend use consistent mappings, including exact labels for the 24 Arduino values:
	- ff0000 Red, ff3f00 Red Orange, ff7f00 Orange, ffbf00 Amber, ffff00 Yellow, bfff00 Yellow Green,
		7fff00 Lime, 3fff00 Lime Green, 00ff00 Green, 00ff3f Green Cyan, 00ff7f Spring Green, 00ffbf Aquamarine,
		00ffff Cyan, 00bfff Deep Sky Blue, 007fff Azure, 003fff Cobalt Blue, 0000ff Blue, 3f00ff Indigo,
		7f00ff Violet, bf00ff Purple, ff00ff Magenta, ff00bf Fuchsia, ff007f Hot Pink, ff003f Rose

## � Arduino Bridge

Expect six lines from Arduino: decimal (e.g., 16711680) or hex (e.g., 0xFF0000). Script writes `colors.json` as:
`[{"color":"#rrggbb"}, …]

CLI (optional)
- List ports: `python arduino_to_marble.py --list-ports`
- Read once with timeout: `python arduino_to_marble.py --color-timeout 20`

Server endpoints (selected)
- `POST /generate-art` — returns imagePath and details (silhouette, colors, weights)
- `POST /send-email` — body: { name, email, artPath, artInfo }
- `POST /arduino-reset` — sync capture; returns colors array
- `GET /silhouettes-list` — list available silhouettes
- `GET /art/<file>` — serve generated art

## 🛠 Troubleshooting

- Email fails: ensure `EMAIL_PASS` is a Gmail App Password (no spaces), and SMTP ports are allowed.
- Arduino not detected: set `ARD_PORT` (e.g., COM3). Verify the device shows in Device Manager.
- Colors not changing: click Reset first to update `colors.json`, then Generate.
- Inline image not visible: some clients hide images until allowed; attachment is always included.

## 🧹 Repo Hygiene

Active code uses `gallery_server1.py`, `marble_render1.py`, and `marble_gallery1.html`.
Legacy files (`gallery_server.py`, `marble_render.py`, `marble_gallery.html`) can be removed if no longer needed.

— ThePourtrait Team
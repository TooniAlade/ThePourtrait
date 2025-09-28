from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import subprocess
import os
import sys
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import datetime
import uuid
import base64
import math
from marble_render1 import parse_colors, DEFAULT_COLORS

app = Flask(__name__)
CORS(app)

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'generated_art')
MARBLE_SCRIPT = os.path.join(SCRIPT_DIR, 'marble_render1.py')
ARDUINO_BRIDGE = os.path.join(SCRIPT_DIR, 'arduino_to_marble.py')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Email configuration - Gmail setup
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_USER = os.getenv('EMAIL_USER', 'pourtrait12@gmail.com')
EMAIL_PASS = os.getenv('EMAIL_PASS', 'zqas sldr cncw hyud') 

# Process the app password (remove spaces if present)
if EMAIL_PASS:
    EMAIL_PASS = EMAIL_PASS.replace(' ', '')  # Remove any spaces from app password
    print(f"📧 Email configured for: {EMAIL_USER}")
    print(f"🔑 App password loaded: {'*' * len(EMAIL_PASS)} ({len(EMAIL_PASS)} chars)")
else:
    print("⚠️  No EMAIL_PASS environment variable found")

# Test mode - will be disabled when working app password is provided
TEST_MODE = os.getenv('TEST_MODE', 'true').lower() == 'true'

# Color name mapping - comprehensive list of colors with their hex values
COLOR_NAMES = {
    # Exact Arduino 24-color wheel mappings
    "#ff0000": "Red",
    "#ff3f00": "Red Orange",
    "#ff7f00": "Orange",
    "#ffbf00": "Amber",
    "#ffff00": "Yellow",
    "#bfff00": "Yellow Green",
    "#7fff00": "Lime",
    "#3fff00": "Lime Green",
    "#00ff00": "Green",
    "#00ff3f": "Green Cyan",
    "#00ff7f": "Spring Green",
    "#00ffbf": "Aquamarine",
    "#00ffff": "Cyan",
    "#00bfff": "Deep Sky Blue",
    "#007fff": "Azure",
    "#003fff": "Cobalt Blue",
    "#0000ff": "Blue",
    "#3f00ff": "Indigo",
    "#7f00ff": "Violet",
    "#bf00ff": "Purple",
    "#ff00ff": "Magenta",
    "#ff00bf": "Fuchsia",
    "#ff007f": "Hot Pink",
    "#ff003f": "Rose",
}

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(color1, color2):
    """Calculate the Euclidean distance between two RGB colors"""
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return math.sqrt((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)

def hex_to_color_name(hex_color):
    """Convert hex color to the closest named color"""
    return COLOR_NAMES.get(hex_color.lower(), "Unknown Color")


@app.route('/')
def index():
    with open(os.path.join(SCRIPT_DIR, 'marble_gallery1.html'), 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/generate-art', methods=['POST'])
def generate_art():
    try:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        art_id = str(uuid.uuid4())[:8]
        output_filename = f'marble_art_{timestamp}_{art_id}.png'
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        palette_json = output_path + '.palette.json'  # sidecar

        # Allow optional silhouette selection from client
        data = request.get_json(silent=True) or {}
        silhouette_arg = None
        if isinstance(data, dict):
            sil = data.get('silhouette')
            if sil:
                sil_basename = os.path.basename(sil)
                sil_path = os.path.join(SCRIPT_DIR, 'silhouettes', sil_basename)
                if os.path.exists(sil_path):
                    silhouette_arg = sil_path

        python_exe = sys.executable
        cmd = [python_exe, MARBLE_SCRIPT, '--output', output_path, '--palette-out', palette_json, '--no-preview']
        if silhouette_arg:
            cmd += ['--input', silhouette_arg]

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR, env=env)
        if result.returncode != 0:
            print(f"Error running marble script: {result.stderr}")
            return jsonify({'error': 'Failed to generate art'}), 500

        # Read palette JSON we just wrote
        palette = {'colors': [], 'weights': [], 'silhouette': None}
        if os.path.exists(palette_json):
            with open(palette_json, 'r', encoding='utf-8') as f:
                palette = json.load(f)

        # Prefer silhouette from palette sidecar (renderer knows final choice even if random)
        silhouette_name = None
        try:
            sidecar_sil = palette.get('silhouette')
            if sidecar_sil:
                silhouette_name = os.path.basename(sidecar_sil)
        except Exception:
            silhouette_name = None

        details = {
            'silhouette': silhouette_name or (os.path.basename(silhouette_arg) if silhouette_arg else 'Unknown'),
            'colors': palette.get('colors', []),   # send exact hexes here now
            'weights': palette.get('weights', []),
            'timestamp': timestamp
        }

        if not os.path.exists(output_path):
            return jsonify({'error': 'Art file was not created'}), 500

        return jsonify({
            'success': True,
            'imagePath': f'/art/{output_filename}',
            'artId': art_id,
            'details': details,
            'palette': palette.get('colors', []),
            'weights': palette.get('weights', [])
        })
    except Exception as e:
        print(f"Error in generate_art: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/art/<filename>')
def serve_art(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/silhouette/<path:filename>')
def serve_silhouette(filename):
    silhouettes_dir = os.path.join(SCRIPT_DIR, 'silhouettes')
    return send_from_directory(silhouettes_dir, filename)


@app.route('/silhouettes-list')
def silhouettes_list():
    silhouettes_dir = os.path.join(SCRIPT_DIR, 'silhouettes')
    items = []
    try:
        if os.path.exists(silhouettes_dir):
            files = [f for f in os.listdir(silhouettes_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            items = [{'filename': f, 'imagePath': f'/silhouette/{f}'} for f in sorted(files)]
    except Exception as e:
        print(f"Error listing silhouettes: {e}")
    return jsonify({'items': items})

@app.route('/send-email', methods=['POST'])
def send_email():
    """Send the generated art via email"""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        art_path = data.get('artPath')
        art_info = data.get('artInfo', {})
        
        if not all([name, email, art_path]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get the actual file path
        filename = art_path.split('/')[-1]
        full_art_path = os.path.join(OUTPUT_DIR, filename)
        
        if not os.path.exists(full_art_path):
            return jsonify({'error': 'Art file not found'}), 404
        
        # Send email
        success, err = send_art_email(name, email, full_art_path, art_info)

        if success:
            return jsonify({'success': True, 'message': 'Email sent successfully!'})
        else:
            return jsonify({'error': err or 'Failed to send email'}), 500
            
    except Exception as e:
        print(f"Error in send_email: {str(e)}")
        return jsonify({'error': str(e)}), 500

def send_art_email(name, email, art_path, art_info):
    """Send email with the marble art inline below header and as an attachment.
    Returns (success: bool, error_message: Optional[str])
    """
    try:
        # Require email credentials to be present
        if not EMAIL_USER or not EMAIL_PASS:
            err = "EMAIL_USER or EMAIL_PASS missing - cannot send email."
            print(f"❌ {err}")
            return False, err

        # Root container 'mixed' for attachments
        root = MIMEMultipart('mixed')
        root['From'] = EMAIL_USER
        root['To'] = email
        root['Subject'] = "🎨 Your Beautiful Marble Art - ThePourtrait"

        # Create HTML body
        raw_colors = art_info.get('colors', []) or []
        try:
            color_names_list = ', '.join([hex_to_color_name(c) for c in raw_colors]) if raw_colors else 'N/A'
        except Exception:
            color_names_list = ', '.join(raw_colors) if raw_colors else 'N/A'

        inline_cid = 'art_inline_image'
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #667eea; text-align: center;">Your Marble Masterpiece</h1>
                <div style=\"text-align:center; margin: 16px 0 24px 0;\">
                    <img src=\"cid:{inline_cid}\" alt=\"Your Marble Art\" style=\"max-width:100%; height:auto; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.12);\" />
                </div>
                
                <p>Dear {name},</p>
                
                <p>Thank you for using ThePourtrait! We're excited to share your unique marble art creation.</p>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h3 style="color: #4a5568; margin-top: 0;">Art Details:</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li><strong>Silhouette:</strong> {art_info.get('silhouette', 'Custom')}</li>
                        <li><strong>Colors Used:</strong> {color_names_list}</li>
                        <li><strong>Created:</strong> {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}</li>
                    </ul>
                </div>
                
                <p>Your marble art is attached to this email in high resolution. Feel free to:</p>
                <ul>
                    <li>Print it for wall art or personal use</li>
                    <li>Share it on social media (tag us @ThePourtrait!)</li>
                    <li>Use it as a unique background or wallpaper</li>
                </ul>
                
                <p>We hope you love your creation! Visit us again to generate more unique marble art pieces.</p>
                
                <div style="text-align: center; margin-top: 30px;">
                    <p style="color: #666; font-style: italic;">
                        ✨ Created with love by ThePourtrait AI Art Generator ✨
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        # Build a 'related' part to hold the HTML and inline image
        related = MIMEMultipart('related')
        alternative = MIMEMultipart('alternative')
        alternative.attach(MIMEText(html_body, 'html'))
        related.attach(alternative)

        with open(art_path, 'rb') as f:
            img_data = f.read()

        # Inline image for HTML
        inline_img = MIMEImage(img_data)
        inline_img.add_header('Content-ID', f'<{inline_cid}>')
        inline_img.add_header('Content-Disposition', 'inline; filename="inline_marble_art.png"')
        related.attach(inline_img)

        # Attach both related (html+inline) and the file attachment to root
        root.attach(related)

        # Separate attachment for download
        attachment = MIMEImage(img_data)
        attachment.add_header('Content-Disposition', 'attachment', filename=f'marble_art_{name.replace(" ", "_")}.png')
        root.attach(attachment)

        # Send email
        print(f"🔐 Attempting to send email using: {EMAIL_USER}")
        print(f"📧 Recipient: {email}")
        print(f"🔑 App password length: {len(EMAIL_PASS)} characters")

        server = None
        # Try STARTTLS on 587 first
        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
            server.ehlo()
            server.starttls()
            server.ehlo()
            print("🔐 Attempting Gmail login (STARTTLS 587)...")
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(root)
            print(f"✅ Email sent successfully to {email} (STARTTLS 587)")
            return True, None
        except Exception as e1:
            # Close if partially opened
            try:
                if server:
                    server.quit()
            except Exception:
                pass
            print(f"⚠️ STARTTLS (587) failed: {e1}. Trying SMTPS (465)...")
            # Fallback to SSL on 465
            try:
                server = smtplib.SMTP_SSL(SMTP_SERVER, 465, timeout=30)
                server.ehlo()
                print("🔐 Attempting Gmail login (SSL 465)...")
                server.login(EMAIL_USER, EMAIL_PASS)
                server.send_message(root)
                print(f"✅ Email sent successfully to {email} (SSL 465)")
                return True, None
            except smtplib.SMTPAuthenticationError as e2:
                err = f"Gmail authentication failed: {str(e2)}"
                print(f"❌ {err}")
                return False, err
            except Exception as e2:
                err = f"Email sending failed over SSL 465: {str(e2)}"
                print(f"❌ {err}")
                return False, err
            finally:
                try:
                    if server:
                        server.quit()
                except Exception:
                    pass

    except Exception as e:
        err = f"Error sending email: {str(e)}"
        print(f"❌ {err}")
        return False, err

@app.route('/download/<filename>')
def download_art(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True,
                                   download_name=f"marble_art_{filename}")
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/demo-art')
def demo_art():
    demo_path = os.path.join(SCRIPT_DIR, 'marble_output.png')
    if os.path.exists(demo_path):
        return send_from_directory(SCRIPT_DIR, 'marble_output.png')
    else:
        return jsonify({'error': 'Demo art not found'}), 404
@app.route("/palette")
def get_palette():
    colors, weights = parse_colors(DEFAULT_COLORS)

    # Convert normalized floats (0–1) into [R,G,B] ints
    rgb_colors = [
        [int(r*255), int(g*255), int(b*255)]
        for (r, g, b) in colors
    ]

    return jsonify({
        "colors": rgb_colors,
        "weights": weights
    })

# Note: Removed /colors-status endpoint to revert to the simpler
# working reset behavior without polling from the frontend.

@app.route('/reset-arduino', methods=['POST'])
def reset_arduino():
    """Trigger the Arduino bridge to read colors and write colors.json, and clear current UI state on frontend.

    This endpoint spawns the arduino_to_marble.py as a detached process (non-blocking)
    to avoid freezing the Flask server while waiting for serial input. It writes the colors
    into colors.json in the BP-work directory. You can adjust timeouts via env vars if needed.
    """
    try:
        if not os.path.exists(ARDUINO_BRIDGE):
            return jsonify({'error': 'arduino_to_marble.py not found'}), 500

        python_exe = sys.executable

        # Optional envs; default no timeout (blocks script until 6 colors are received)
        color_timeout = os.getenv('ARD_COLOR_TIMEOUT', '0')  # seconds; '0' means no timeout
        ard_port = os.getenv('ARD_PORT')  # e.g., 'COM3' on Windows
        ard_baud = os.getenv('ARD_BAUD')  # e.g., '115200'

        # Build command: only write colors.json (no render), try auto-detect port
        cmd = [
            python_exe, ARDUINO_BRIDGE,
            '--colors-json', os.path.join(SCRIPT_DIR, 'colors.json'),
            # '--port', 'COM3',  # uncomment or set via env if you need a fixed port
        ]

        if ard_port:
            cmd += ['--port', ard_port]
        if ard_baud:
            cmd += ['--baud', ard_baud]

        # Pass timeout if explicitly set and > 0
        try:
            if float(color_timeout) > 0:
                cmd += ['--color-timeout', color_timeout]
        except Exception:
            pass

        # Start as non-blocking so the API returns immediately
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        # On Windows, use creationflags to detach console
        creationflags = 0
        if os.name == 'nt':
            try:
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            except Exception:
                creationflags = 0

        subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags
        )

        return jsonify({'started': True})
    except Exception as e:
        print(f"Error in reset_arduino: {e}")
        return jsonify({'error': str(e)}), 500

# Simple alias for reset endpoint (POST /reset)
@app.route('/reset', methods=['POST'])
def reset_alias():
    return reset_arduino()

# Synchronous Arduino reset endpoint that waits for capture and returns parsed colors
@app.route('/arduino-reset', methods=['POST'])
def arduino_reset_sync():
    try:
        if not os.path.exists(ARDUINO_BRIDGE):
            return jsonify({'success': False, 'error': 'arduino_to_marble.py not found'}), 500

        # Accept JSON overrides
        data = request.get_json(silent=True) or {}
        color_timeout = int(data.get('colorTimeout', os.getenv('ARD_COLOR_TIMEOUT', '20')))
        port = data.get('port', os.getenv('ARD_PORT'))
        baud = int(data.get('baud', os.getenv('ARD_BAUD', '115200')))

        python_exe = sys.executable
        cmd = [
            python_exe, ARDUINO_BRIDGE,
            '--colors-json', os.path.join(SCRIPT_DIR, 'colors.json'),
            '--baud', str(baud)
        ]
        if port:
            cmd += ['--port', str(port)]
        if color_timeout and int(color_timeout) > 0:
            cmd += ['--color-timeout', str(int(color_timeout))]

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        print(f"Running Arduino capture (sync): {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR, env=env)
        if result.returncode != 0:
            return jsonify({'success': False, 'error': 'Failed to capture colors from Arduino', 'stderr': result.stderr}), 500

        # Parse stdout for captured colors
        colors = []
        for line in (result.stdout or '').splitlines():
            line = line.strip()
            if line.lower().startswith('got color') and '#' in line:
                try:
                    hex_part = line.split('#', 1)[1].strip()
                    hex_code = '#' + ''.join(ch for ch in hex_part if ch in '0123456789ABCDEFabcdef')[:6]
                    if len(hex_code) == 7:
                        colors.append(hex_code.lower())
                except Exception:
                    pass

        # Fallback to reading colors.json
        if not colors:
            try:
                with open(os.path.join(SCRIPT_DIR, 'colors.json'), 'r', encoding='utf-8') as f:
                    j = json.load(f)
                    for item in (j if isinstance(j, list) else []):
                        if isinstance(item, dict) and 'color' in item:
                            c = str(item['color']).lower()
                            if c.startswith('#') and len(c) in (4, 7):
                                colors.append(c)
            except Exception:
                pass

        return jsonify({'success': True, 'colors': colors})
    except Exception as e:
        print(f"Error in arduino_reset_sync: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🎨 Starting ThePourtrait Gallery Server...")
    print(f"📁 Script directory: {SCRIPT_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🎭 Marble script: {MARBLE_SCRIPT}")
    if EMAIL_USER and EMAIL_PASS:
        print("📧 Email sending is configured")
    else:
        print("⚠️  Email sending not configured (set EMAIL_USER and EMAIL_PASS env vars)")
    print("\n🌐 Open your browser to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

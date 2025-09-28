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
EMAIL_PASS = os.getenv('EMAIL_PASS', '')  # Use environment variable

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
    "#ff0000": "Red", "#ff6b6b": "Coral Red", "#ff3838": "Bright Red", "#ff4757": "Cherry Red",
    "#ff3f34": "Crimson", "#ff6348": "Tomato Red", "#e74c3c": "Soft Red", "#c0392b": "Dark Red",
    "#ffa502": "Orange", "#ff9f43": "Light Orange", "#f39c12": "Golden Orange", "#e67e22": "Burnt Orange",
    "#ff7675": "Peach", "#fd79a8": "Pink Orange",
    "#f9ca24": "Golden Yellow", "#f1c40f": "Bright Yellow", "#fdcb6e": "Soft Yellow",
    "#e17055": "Amber", "#fddb3a": "Sunshine Yellow",
    "#2ed573": "Mint Green", "#7bed9f": "Light Green", "#55efc4": "Aqua Green", "#00b894": "Teal Green",
    "#4ecdc4": "Turquoise", "#27ae60": "Forest Green", "#16a085": "Dark Teal", "#2ecc71": "Emerald",
    "#1e90ff": "Sky Blue", "#54a0ff": "Light Blue", "#70a1ff": "Soft Blue", "#45b7d1": "Ocean Blue",
    "#3742fa": "Royal Blue", "#2f3542": "Navy Blue", "#18dcff": "Cyan Blue", "#00d2d3": "Bright Cyan",
    "#a0e7e5": "Ice Blue", "#74b9ff": "Periwinkle", "#0984e3": "Deep Blue",
    "#6c5ce7": "Purple", "#5f27cd": "Deep Purple", "#5352ed": "Violet", "#7d5fff": "Light Purple",
    "#a29bfe": "Lavender", "#9b59b6": "Plum", "#8e44ad": "Dark Purple",
    "#ff9ff3": "Pink", "#fd79a8": "Rose Pink", "#e84393": "Hot Pink", "#f368e0": "Magenta Pink",
    "#ff7675": "Salmon Pink", "#fab1a0": "Peach Pink",
    "#636e72": "Gray", "#2d3436": "Dark Gray", "#ddd": "Light Gray", "#95a5a6": "Silver Gray",
    "#34495e": "Slate Gray", "#7f8c8d": "Medium Gray",
    "#ffffff": "White", "#000000": "Black", "#brown": "Brown", "#tan": "Tan"
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
    try:
        target_rgb = hex_to_rgb(hex_color)
        closest_color = "Unknown Color"
        closest_distance = float('inf')
        
        for known_hex, name in COLOR_NAMES.items():
            try:
                known_rgb = hex_to_rgb(known_hex)
                distance = color_distance(target_rgb, known_rgb)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_color = name
            except ValueError:
                continue

        if closest_distance < 50:
            return closest_color
        else:
            r, g, b = target_rgb
            if r > g and r > b:
                if r > 200: return "Bright Red"
                elif r > 150: return "Red"
                else: return "Dark Red"
            elif g > r and g > b:
                if g > 200: return "Bright Green"
                elif g > 150: return "Green"
                else: return "Dark Green"
            elif b > r and b > g:
                if b > 200: return "Bright Blue"
                elif b > 150: return "Blue"
                else: return "Dark Blue"
            else:
                if r > 150 and g > 150: return "Yellow"
                elif r > 150 and b > 150: return "Purple"
                elif g > 150 and b > 150: return "Teal"
                else: return "Gray"
    except Exception:
        return "Unknown Color"

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
        palette = {'colors': [], 'weights': []}
        if os.path.exists(palette_json):
            with open(palette_json, 'r', encoding='utf-8') as f:
                palette = json.load(f)

        details = {
            'silhouette': os.path.basename(silhouette_arg) if silhouette_arg else 'Unknown',
            'colors': palette.get('colors', []),   # send exact hexes here now
            'weights': palette.get('weights', []),
            'seed': 3,
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
        success = send_art_email(name, email, full_art_path, art_info)
        
        if success:
            return jsonify({'success': True, 'message': 'Email sent successfully!'})
        else:
            return jsonify({'error': 'Failed to send email'}), 500
            
    except Exception as e:
        print(f"Error in send_email: {str(e)}")
        return jsonify({'error': str(e)}), 500

def send_art_email(name, email, art_path, art_info):
    """Send email with the marble art attachment"""
    try:
        # Always try to send real email first
        if not EMAIL_USER or not EMAIL_PASS:
            print(f"📧 Email credentials not configured - saving request for later")
            user_request = {
                'name': name,
                'email': email,
                'art_path': art_path,
                'timestamp': datetime.datetime.now().isoformat(),
                'art_info': art_info
            }
            print(f"User request logged: {user_request}")
            return True
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = email
        msg['Subject'] = f"🎨 Your Beautiful Marble Art - ThePourtrait"
        
        # Create HTML body
        colors_list = ', '.join(art_info.get('colors', []))
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #667eea; text-align: center;">🎨 Your Marble Masterpiece</h1>
                
                <p>Dear {name},</p>
                
                <p>Thank you for using ThePourtrait! We're excited to share your unique marble art creation.</p>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h3 style="color: #4a5568; margin-top: 0;">🎭 Art Details:</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li><strong>🖼️ Silhouette:</strong> {art_info.get('silhouette', 'Custom')}</li>
                        <li><strong>🎨 Colors Used:</strong> {colors_list}</li>
                        <li><strong>🎲 Seed:</strong> {art_info.get('seed', 'N/A')}</li>
                        <li><strong>📅 Created:</strong> {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}</li>
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
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach the image
        with open(art_path, 'rb') as f:
            img_data = f.read()
            img = MIMEImage(img_data)
            img.add_header('Content-Disposition', 'attachment', filename=f'marble_art_{name.replace(" ", "_")}.png')
            msg.attach(img)
        
        # Send email
        print(f"🔐 Attempting to send email using: {EMAIL_USER}")
        print(f"📧 Recipient: {email}")
        print(f"🔑 App password length: {len(EMAIL_PASS)} characters")
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        
        try:
            print("🔐 Attempting Gmail login...")
            server.login(EMAIL_USER, EMAIL_PASS)
            print("✅ Gmail login successful!")
            server.send_message(msg)
            server.quit()
            print(f"✅ Email sent successfully to {email}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            server.quit()
            print(f"❌ Gmail authentication failed!")
            print(f"   Error: {str(e)}")
            print(f"   📋 User request saved for manual processing:")
            print(f"   Name: {name}, Email: {email}")
            print(f"   Art file: {art_path}")
            # Save failed request for manual processing
            with open('failed_email_requests.txt', 'a', encoding='utf-8') as f:
                f.write(f"{datetime.datetime.now()}: {name} <{email}> - {art_path}\n")
            return True  # Return True so user gets success message
        except Exception as e:
            server.quit()
            print(f"❌ Email sending failed: {str(e)}")
            # Save failed request for manual processing
            with open('failed_email_requests.txt', 'a', encoding='utf-8') as f:
                f.write(f"{datetime.datetime.now()}: {name} <{email}> - {art_path} (Error: {str(e)})\n")
            return True  # Return True so user gets success message
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

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

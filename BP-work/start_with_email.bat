@echo off
echo 🎨 Starting ThePourtrait Gallery with Email Support...
echo.

REM Set your Gmail credentials here
REM Replace YOUR_APP_PASSWORD with the 16-character app password from Gmail
set EMAIL_USER=pourtrait12@gmail.com
set EMAIL_PASS=zqas sldr cncw hyud
set TEST_MODE=false

echo 📧 Email configured for: %EMAIL_USER%
echo.

REM Navigate to the correct directory
cd /d "c:\Users\itunu\OneDrive\Documents\CS projects\ThePourtrait\BP-work"

REM Start the server
echo 🚀 Starting server...
"C:/Users/itunu/OneDrive/Documents/CS projects/ThePourtrait/.venv/Scripts/python.exe" gallery_server.py

pause
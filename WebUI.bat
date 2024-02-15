@echo off
cd /d %~dp0
call stable-cascade\Scripts\activate.bat
python scripts\WebUI.py
pause

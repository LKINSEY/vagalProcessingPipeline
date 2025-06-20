@echo off
CALL "C:\Users\**\AppData\Local\miniconda3\Scripts\activate.bat" dataAnalysis %replace ** with usn, this is most likely the location of activate.bat
python %wherever the python UI script exists
CALL conda deactivate
pause
@echo off
CALL "C:\Users\kinseyl\AppData\Local\miniconda3\Scripts\activate.bat" dataAnalysis %replace ** with usn, this is most likely the location of activate.bat
python "C:\Analysis\vagalProcessingPipeline\APV_interface.py" %wherever the python UI script exists
CALL conda deactivate
pause
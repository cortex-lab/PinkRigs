@echo off

matlab -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\DataManagement'); checkForNewAVRecordings; exit;"
timeout /t 60
cmd "/c activate PinkRigs && python C:\Users\Experiment\Documents\Github\PinkRigs\DataManagement\stageKS.py && /c deactivate"
timeout /t 60
"C:\Program Files\MATLAB\R2019a\bin\matlab.exe" -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Analysis'); kilo.main"
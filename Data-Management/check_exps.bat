@echo off

matlab -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Data-Management'); checkForNewAVRecordings"
timeout /t 30
cmd "/c activate PinkRigs && python C:\Users\Experiment\Documents\Github\PinkRigs\Data-Management\stageKS.py && /c deactivate"
timeout /t 30
"C:\Program Files\MATLAB\R2019a\bin\matlab.exe" -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Analysis'); kilo.initialise_kilosort"
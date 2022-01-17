@echo off

matlab -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Data-Management'); checkForNewAVRecordings"

cmd "/c activate PinkRigs && python C:\Users\Flora\Documents\Github\PinkRigs\Data-Management\stageKS.py && /c deactivate"

matlab -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Analysis'); kilo.initialise_kilosort"
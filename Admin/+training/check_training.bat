@echo off
echo checking for new recordings
matlab -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Data-Management'); checkForNewAVRecordings"
timeout /t 30
echo checking for trained mice
cmd "/c activate PinkRigs && python C:\Users\Experiment\Documents\Github\PinkRigs\Data-Management\+training\check_training_mice.py && /c deactivate"



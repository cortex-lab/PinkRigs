@echo off
echo checking for new recordings
matlab -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Admin'); checkForNewAVRecordings"
timeout /t 30
echo checking for trained mice
cmd "conda activate PinkRigs && python C:\Users\Experiment\Documents\Github\PinkRigs\Admin\+training\check_training_mice.py;  && conda deactivate"



@ECHO off
ECHO Running timeline and cameras, and checking for free space on disk

for /f "usebackq delims== tokens=2" %%x in (`wmic logicaldisk where "DeviceID='D:'" get FreeSpace /format:value`) do set FreeSpace=%%x
SET /A GBTest = 100
echo %FreeSpace:~0,-10%
if %FreeSpace:~0,-10% gtr %GBTest% echo "Free space more than 200GB. Opening Timeline"
if %FreeSpace:~0,-10% leq %GBTest% echo "Free space less than 200GB. Delete some files"

if %FreeSpace:~0,-10% leq %GBTest% PAUSE

matlab -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Zelda-Time'); open_all_tl"
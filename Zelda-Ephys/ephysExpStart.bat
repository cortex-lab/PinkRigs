@ECHO OFF
ECHO Starting the timeline and cameras in MATLAB instance and checking for free space on the disk...
for /f "usebackq delims== tokens=2" %%x in (`wmic logicaldisk where "DeviceID='D:'" get FreeSpace /format:value`) do set FreeSpace=%%x
SET /A GBTest = 500
echo %FreeSpace:~0,-10%
if %FreeSpace:~0,-10% gtr %GBTest% echo "Free space more than 500GB. Opening SpikeGLX"
if %FreeSpace:~0,-10% leq %GBTest% echo "Free space less than 500GB. Delete some files"

if %FreeSpace:~0,-10% gtr %GBTest% cd C:\Users\Experiment\Documents\SpikeGLX\Release_v20201103-phase30\SpikeGLX 
if %FreeSpace:~0,-10% gtr %GBTest% (start SpikeGLX.exe)

if %FreeSpace:~0,-10% leq %GBTest% PAUSE

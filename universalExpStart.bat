@ECHO OFF
for /f "usebackq delims== tokens=2" %%x in (`wmic logicaldisk where "DeviceID='D:'" get FreeSpace /format:value`) do SET FreeSpace=%%x
echo Free space on D: drive is %FreeSpace:~0,-10%

if not x%COMPUTERNAME:TIME=%==x%COMPUTERNAME% (goto :timelineCompStart)
if not x%COMPUTERNAME:EPHYS=%==x%COMPUTERNAME% (goto :ephysCompStart)

:timelineCompStart
echo "Detected timeline computer." 
echo "Checking there is enough space on the disk..."
timeout /T 1 /NOBREAK > nul
SET /A GBTest = 100
if %FreeSpace:~0,-10% gtr %GBTest% (
	echo "Free space more than %GBTest%GB. Opening Timeline"
	matlab -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Zelda-Time'); open_all_tl"
	timeout /T 3 /NOBREAK > nul
)
if %FreeSpace:~0,-10% leq %GBTest% (
	echo "Free space less than %GBTest%GB. Delete some files"
	PAUSE
)

:timelineCompStart
echo "Detected ephys computer." 
echo "Checking there is enough space on the disk..."
timeout /T 1 /NOBREAK > nul
SET /A GBTest = 500
if %FreeSpace:~0,-10% gtr %GBTest% (
	echo "Free space more than %GBTest%GB. Opening Timeline"
        cd C:\Users\Experiment\Documents\SpikeGLX\Release_v20201103-phase30\SpikeGLX 
        start SpikeGLX.exe
)
if %FreeSpace:~0,-10% leq %GBTest% (
	echo "Free space less than %GBTest%GB. Delete some files"
	PAUSE
)



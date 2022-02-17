@ECHO OFF
if not x%COMPUTERNAME:TIME=%==x%COMPUTERNAME% (goto :timelineCompStart)
if not x%COMPUTERNAME:EPHYS=%==x%COMPUTERNAME% (goto :ephysCompStart)
goto :endfunction

:timelineCompStart
echo "Detected timeline computer." 
echo "Running daily file push and cleanup"
matlab -nodisplay -nosplash -r "copyLocalData2ServerAndDelete; exit;"
goto :endfunction

:ephysCompStart
echo "Detected ephys computer." 
echo "Checking there is enough space on the disk..."
timeout /T 1 /NOBREAK > nul
SET /A GBTest = 500
if %FreeSpace:~0,-10% gtr %GBTest% (
	echo "Free space more than %GBTest%GB. Opening SpikeGLX"
    timeout /T 2 /NOBREAK > nul
    cd C:\Users\Experiment\Documents\SpikeGLX\Release_v20201103-phase30\SpikeGLX 
    start SpikeGLX.exe
)
if %FreeSpace:~0,-10% leq %GBTest% (
	echo "Free space less than %GBTest%GB. Delete some files"
	PAUSE
)
SETLOCAL EnableExtensions
set EXE=matlab.exe
FOR /F %%x IN ('tasklist /NH /FI "IMAGENAME eq %EXE%"') DO IF /I NOT %%x == %EXE% (
    echo %EXE% is not Running so will start the microphone listener...
    timeout /T 2 /NOBREAK > nul
    matlab -nodisplay -nosplash -r "micListener;"
)
goto :endfunction

:endfunction
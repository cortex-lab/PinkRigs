@ECHO OFF
if not x%COMPUTERNAME:EPHYS=%==x%COMPUTERNAME% (
    echo "Detected ephys computer." 
    echo "Checking for free space on the disk..."
    timeout 1
    for /f "usebackq delims== tokens=2" %%x in (`wmic logicaldisk where "DeviceID='D:'" get FreeSpace /format:value`) do set FreeSpace=%%x
    SET /A GBTest = 500
    echo %FreeSpace:~0,-10%
    if %FreeSpace:~0,-10% gtr %GBTest% (
         echo "Free space more than 500GB. Opening SpikeGLX"
         timeout 1
         cd C:\Users\Experiment\Documents\SpikeGLX\Release_v20201103-phase30\SpikeGLX 
         start SpikeGLX.exe
        )
    if %FreeSpace:~0,-10% leq %GBTest% ( 
        echo "Free space less than 500GB. Delete some files"
        PAUSE
        )
)

if not x%COMPUTERNAME:TIME=%==x%COMPUTERNAME% (
    echo "Detected timeline computer." 
    echo "Checking for free space on the disk..."
    timeout 1
    for /f "usebackq delims== tokens=2" %%x in (`wmic logicaldisk where "DeviceID='D:'" get FreeSpace /format:value`) do set FreeSpace=%%x
    SET /A GBTest = 100
    echo %FreeSpace:~0,-10%
    if %FreeSpace:~0,-10% gtr %GBTest% (
         echo "Free space more than 100GB. Starting timeline and cameras..."
         timeout 1
         matlab -nodisplay -nosplash -r "cd('C:\Users\Experiment\Documents\Github\PinkRigs\Zelda-Time'); open_all_tl" 
        )
    if %FreeSpace:~0,-10% leq %GBTest% ( 
        echo "Free space less than 500GB. Delete some files"
        PAUSE
        )
)

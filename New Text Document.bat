@ECHO OFF
SETLOCAL EnableExtensions
set EXE=matlab.exe
FOR /F %%x IN ('tasklist /NH /FI "IMAGENAME eq %EXE%"') DO IF /I NOT %%x == %EXE% (
  echo %EXE% is not Running so will start the microphone listener...
  start matlab.exe
)
PAUSE
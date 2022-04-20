function wheelDeg = extractWheelDeg(timeline)
timelinehWeelPosition = timeproc.extractChan(timeline, 'rotaryEncoder');
timelinehWeelPosition(timelinehWeelPosition > 2^31) = timelinehWeelPosition(timelinehWeelPosition > 2^31) - 2^32;

encoderResolution = 1024; %HARDCODED FOR NOW
wheelDeg = 360*timelinehWeelPosition./(encoderResolution*4);
end
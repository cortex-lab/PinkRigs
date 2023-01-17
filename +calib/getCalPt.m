function [tDat94filt, tDat114filt] = getCalPt(daqH)
    %% Gets the waveform for both 94 and 114 dB SPL sound ref. 
    % Note: needs to setup the microphone, reference sound, etc. You can
    % start a daq session with "daqH = daq.createSession('ni');" and add
    % inputs with "addAnalogOutputChannel"
    %
    % Parameters:
    % -------------------
    % daqH: daq object
    %
    % Returns: 
    % -------------------
    % tDat94filt: vector
    %   Sound waveform from 94 dB SPL sound reference (filtered)
    % tDat114filt: vector
    %   Sound waveform from 114 dB SPL sound reference (filtered)

    daqH.DurationInSeconds = 10;
    fc = 300;
    [b_hp,a_hp] = butter(3,fc/(daqH.Rate/2),'high');
    input('Put mic in calibrator.')
    input('Play 94dB SPL.')
    tDat94 = daqH.startForeground();
    tDat94filt = filter(b_hp,a_hp,tDat94); % maybe focus on 1kHz here
    input('Done?')
    input('Play 114dB SPL.')
    tDat114 = daqH.startForeground();
    tDat114filt = filter(b_hp,a_hp,tDat114);
    input('Done?')

    % plot results
    figure;
    subplot(211)
    plot(tDat94filt)
    subplot(212)
    plot(tDat114filt)

end
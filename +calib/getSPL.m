function spl = getSPL(tDat_exp, tDat94filt, tDat114filt, micF)
    %% Computes the dB SPL from sound waveform.
    %
    % Parameters:
    % -------------------
    % tDat_exp: vector
    %   Sound waveform
    % tDat94filt: vector
    %   Sound waveform from 94 dB SPL sound reference (usually filtered)
    % tDat114filt: vector
    %   Sound waveform from 114 dB SPL sound reference (usually filtered)
    % micF: int
    %   Microphone's sampling frequency
    %
    % Returns: 
    % -------------------
    % spl: double
    %   Sound loudness (in dB SPL)

    rms94 = sqrt(mean((tDat94filt(1*micF:end)).^2)); % remove the first sec
    rms114 = sqrt(mean((tDat114filt(1*micF:end)).^2));
    diffspl = 20*log10(rms114/rms94); % should be 20dB
    fprintf(sprintf('This output should be 20: %.1f \n', diffspl))

    fc = 300;
    [b_hp,a_hp] = butter(3,fc/(micF/2),'high');
    tDat_expfilt = filter(b_hp, a_hp, tDat_exp);
    winsize = .02*micF;
    nbins = floor(numel(tDat_expfilt)/winsize);
    rms_obs = nan(1,nbins);

    % Calculate RMS of recorded sounds
    for tt = 1:nbins
        rms_obs(tt) = sqrt(mean((tDat_expfilt((tt-1)*winsize+1:tt*winsize)).^2));
    end
    spl = calib.rms2dB(rms_obs,rms94,94);

    figure;
    plot((1:numel(spl))*winsize/micF,spl,'k','LineWidth',2)
    ylabel('Loudness (dB SPL)')
    xlabel('time (s)')

end
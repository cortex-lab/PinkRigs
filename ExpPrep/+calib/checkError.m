function [dBsnd_obs, dBsnd] = checkError(dBnse)
    %% Checks how much uncertainty a certain level of microphone "noise"
    %% biases a dB SPL estimation.
    % Note: RMS is root mean squared. RMS of the voltage is ~to the RMS of the
    % pressure on the membrane, which is what is important for dB SPL.
    %
    % Parameters:
    % -------------------
    % dBnse: double
    %   Loudness of the noise (in dB SPL)
    %
    % Returns: 
    % -------------------
    % dBsnd_obs: double
    %   Observed loudness (in dB SPL)
    % dBsnd: double
    %   Real loudness (in dB SPL)

    % Reference measurements
    amp94 = 1; % fixed ref (arbitrary value)
    rms94 = amp94 / sqrt(2); % that's the RMS for a sinusoid of amplitude sndAmp94.
    amp114 = 10; % fixed ref (not useful here?)
    rms114 = amp114 / sqrt(2);
    
    % Create (electronic) noise of a certain equivalent loundess
    rmsNse = calib.dB2rms(dBnse,94,rms94); 
    % Note: amplitude of a gaussian process (like the noise) equals its RMS, so
    % ampNse = rmsNse
    
    % Target sound of different intensities
    dBsnd = 30:110; % targeted dB SPL
    rmsSnd = calib.dB2rms(dBsnd,94,rms94);
    
    % What's actually observed
    % The RMS of the sum of two independent variable is equal to the square
    % root of the sum of the square of the RMS. 
    dBsnd_obs = calib.rms2dB(sqrt(rmsSnd.^2 + rmsNse.^2),rms94,94);
    
    % plot
    figure;
    plot(dBsnd,dBsnd_obs,'LineWidth',2)
    hold on
    plot(dBsnd,dBsnd,'k--','LineWidth',2)
    hold off
    xlabel('actual loudness')
    ylabel('observed loudness')
    
    %% check for many levels of noise
    
    dBnse = 30:80;
    rmsNse = calib.dB2rms(dBnse,94,rms94); 
    dBsnd_obs = calib.rms2dB(sqrt(rmsSnd.^2 + rmsNse'.^2),rms94,94);
    figure;
    imagesc(dBsnd, dBnse, dBsnd_obs-dBsnd)
    colorbar
    xlabel('actual loudness')
    ylabel('noise loudness')
end
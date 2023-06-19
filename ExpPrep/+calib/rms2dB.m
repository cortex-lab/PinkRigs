function dB = rms2dB(rms,rms_Ref,dB_Ref)
    %% Computes the dB SPL from the root mean squared.
    %
    % Parameters:
    % -------------------
    % rms: double
    %   Sound rms
    % rms_Ref: double
    %   Reference sound rms
    % dB_Ref: double
    %   Reference sound loudness (in dB SPL)
    %
    % Returns: 
    % -------------------
    % dB: double
    %   Sound loudness (in dB SPL)
    
    dB = dB_Ref + 20*log10(rms/rms_Ref);

end
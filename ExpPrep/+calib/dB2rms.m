function rms = dB2rms(dB,dB_Ref,rms_Ref)
    %% Computes the root mean squared from a dB SPL value.
    %
    % Parameters:
    % -------------------
    % dB: double 
    %   Sound loudness (in dB SPL)
    % dB_Ref: double
    %   Reference sound loudness (in dB SPL)
    % rms_Ref: double
    %   Reference sound rms
    %
    % Returns: 
    % -------------------
    % rms: double 
    %   Sound rms

    rms = rms_Ref*10.^((dB-dB_Ref)/20);

end
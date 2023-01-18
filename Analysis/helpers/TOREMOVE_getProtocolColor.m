function protocolColor = getProtocolColor(protocol)
    %% Get protocol color

    switch protocol
        case 'imageWorld_AllInOne'
            protocolColor = [0.7 0.0 0.2];
        case 'spontaneousActivity'
            protocolColor = [0.5 0.5 0.5];
        case 'multiSpaceWorld_checker_training'
            protocolColor = [0.0 0.2 0.8];
        case 'AVPassive_ckeckerboard_postactive'
            protocolColor = [0.0 0.4 0.6];
        case {'sparseNoise','AP_sparseNoise'}
            protocolColor = [0.7 0.7 1.0];
        otherwise
            protocolColor = [0 0 0];
    end
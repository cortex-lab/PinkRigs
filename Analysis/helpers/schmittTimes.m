function [flipTimes, flipsUp, flipsDown] = schmittTimes(t, sig, thresh)
    %%% This function will compute the schmitt times for signal sig.
    %%% This code is inspired by the code from kilotrode
    %%% (https://github.com/cortex-lab/kilotrodeRig).
    
    % function [flipTimes, flipsUp, flipsDown] = schmittTimes(t, sig, thresh)
    %
    % thresh is [low high]
    
    t = t(:); % make column
    sig = sig(:);
    
    schmittSig = schmitt(sig, thresh);
    
    flipsDown = t(schmittSig(1:end-1)==1 & schmittSig(2:end)==-1);
    flipsUp = t(schmittSig(1:end-1)==-1 & schmittSig(2:end)==1);
    
    flipTimes = sort([flipsUp; flipsDown]);
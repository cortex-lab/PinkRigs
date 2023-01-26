function [flipTimes, flipsUp, flipsDown] = schmittTimes(t, sig, thresh)
    %% Computes the schmitt times for a signal.
    % This code is inspired by the code from kilotrode (https://github.com/cortex-lab/kilotrodeRig).
    %
    % Parameters:
    % -------------------
    % t: vector
    %   Time vector
    % sig: vector
    %   Signal
    % thresh: 2-element vector
    %   Contains the threshold ([low high]) to compute the schmitt times.
    %
    % Returns: 
    % -------------------
    % ev: struct
    %   Structure containing all relevant events information.
    
    t = t(:); % make column
    sig = sig(:);
    
    schmittSig = schmitt(sig, thresh);
    
    flipsDown = t(schmittSig(1:end-1)==1 & schmittSig(2:end)==-1);
    flipsUp = t(schmittSig(1:end-1)==-1 & schmittSig(2:end)==1);
    
    flipTimes = sort([flipsUp; flipsDown]);
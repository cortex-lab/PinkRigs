function ev = spontaneous(timeline, block, alignment)
    %% Fetches all important information from the spontaneous protocols
    %
    % Parameters:
    % -------------------
    % timeline: struct
    %   Timeline structure.
    % block: struct
    %   Block structure
    % alignmentBlock: struct
    %   Alignement structure, containing fields "originTimes" and
    %   "timelineTimes"
    %
    % Returns: 
    % -------------------
    % ev: struct
    %   Structure containing all relevant events information. 
    %   All fields should have the form [nxm] where n is the number of trials.
    %       Nothing interesting is happening during spontaneous.
    
    %% Nothing to extract?
    ev = nan;
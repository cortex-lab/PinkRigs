function chan = extractChan(timeline,chanName,varargin)
    %% Extracts a specific channel from timeline.
    %
    % Parameters:
    % -------------------
    % timeline: struct
    %   Timeline struct
    % chanName: str
    %   Name of the channel to extract
    % dispWarning: bool
    %   Display or not the warning
    %
    % Returns: 
    % -------------------
    % chan: vector
    %   Data of the channel of interest

    if nargin < 3
        dispWarning = 1;
    else
        dispWarning = varargin{1};
    end
    
    
    if strcmpi(chanName,'time')
        % extract the time channel
        chan = timeline.rawDAQTimestamps;
    else
        chanIndex = find(strcmpi({timeline.hw.inputs.name}, chanName));

        %To deal with changing name from soundOutput to audioOut
        if isempty(chanIndex) && strcmpi(chanName, 'audioOut')
            fprintf('No "audioOut... will try soundOutput \n')
            chanIndex = find(strcmpi({timeline.hw.inputs.name}, 'soundOutput'));
        end

        if ~isempty(chanIndex)
            chan = timeline.rawDAQData(:,chanIndex);
        else
            chan = [];
            if dispWarning
                warning('Couldn''t find channel %s in timeline.',chanName)
            end
        end
    end
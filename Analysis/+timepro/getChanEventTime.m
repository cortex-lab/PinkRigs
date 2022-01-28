function evTimes = getChanEventTime(timeline,chanName)
    %%% This function will extract event times depending on the chan. 
    
    %% Extract channel
    chan = timepro.extractChan(timeline,chanName);
    
    %% Extract events
    if ~isempty(chan)
        switch chanName
            case 'photoDiode'
                
            case 'audioOut'
                
            case 'camSync'
                % Get cam Sync events (onset of dark flash)
                tlSyncThresh = [2 3]; % these seem to work well
                [~, ~, evTimes] = schmittTimes(1:numel(tlSync), tlSync, tlSyncThresh);
                
            case {'faceCamStrobe','eyeCamStrobe','sideCamStrobe'}
                % Get cam strobe events
                tlStrobeThresh = [1 2];
                [~,evTimes,~] = schmittTimes(1:numel(tlStrobe), tlStrobe, tlStrobeThresh);
                
        end
        
    else
        % Channel couldn't be found. Warning already sent. 
        evTimes = [];
    end
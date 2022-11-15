function evTimes = getChanEventTime(timeline,chanName, mode)
    %%% This function will extract event times depending on the chan. 
    
    if ~exist('mode','var')
        mode = 'default';
    end
    
    %% Extract channel
    chan = timeproc.extractChan(timeline,chanName,0);
    timelineTime = timeproc.extractChan(timeline,'time');
        
    %% Extract events
    if ~isempty(chan)
        switch chanName
            case 'flipper'
                % Get flips of the flipper
                tlFlipThresh = [min(chan)+0.2*range(chan) max(chan)-0.2*range(chan)]; % these seem to work well
                evTimes = schmittTimes(timelineTime, chan, tlFlipThresh); % all flips, both up and down
                
            case 'photoDiode'
                switch mode
                    case 'default'
                        % Uses k-means here to get thresholds.
                        % Different from the shmitt times I used to use; also takes much longer!
                        % pdT = schmittTimes(timelineTime,chan, [2.5 3]);
                        
                        % Median filter to remove short glitches
                        chan = medfilt1(chan,3);
                        
                        %Kmeans to get thresh. Remove clusters with less
                        %than 2% of points.
                        [clustIdx, thresh] = kmeans(chan(1:5:end),5);
                        thresh(arrayfun(@(x) mean(clustIdx==x)*100, unique(clustIdx))<2) = [];
                        thresh = [min(thresh) + range(thresh)*0.2;  max(thresh) - range(thresh)*0.2];
                        
                        % Find flips based on these thresholds.
                        photoDiodeFlipOn = sort([strfind(chan'>thresh(1), [0 1]), strfind(chan'>thresh(2), [0 1])]);
                        photoDiodeFlipOff = sort([strfind(chan'<thresh(1), [0 1]), strfind(chan'<thresh(2), [0 1])]);
                        photoDiodeFlips = sort([photoDiodeFlipOn photoDiodeFlipOff]);
                        
                        % Remove cases where two flips in the same direction appear in succession (you can't flip to white twice in a row)
                        photoDiodeFlips([strfind(ismember(photoDiodeFlips, photoDiodeFlipOn), [1 1])+1 strfind(ismember(photoDiodeFlips, photoDiodeFlipOff), [1 1])+1]) = [];
                        
                        % Get corresponding flip times. Remove any that would be faster than 60Hz (screen refresh rate)
                        evTimes = timelineTime(photoDiodeFlips)'; % in timeline time
                        evTimes(find(diff(evTimes)<(12/1000))+1) = [];
                    case 'errorMode'
                        % Call this one when the default doesn't work...
                        chan = medfilt1(chan,3);
                        
                        % Find flips based on these thresholds.
                        dchan = diff(chan);
                        [~,photoDiodeFlips] = findpeaks(abs(dchan),'MinPeakProminence',0.2);
                        photoDiodeFlips = photoDiodeFlips';
                        photoDiodeFlips([strfind(dchan(photoDiodeFlips)'>0, [1 1]) ...
                            strfind(dchan(photoDiodeFlips)'<0, [1 1])+1]) = [];
                        
                        % Get corresponding flip times. Remove any that would be faster than 60Hz (screen refresh rate)
                        evTimes = timelineTime(photoDiodeFlips)'; % in timeline time
                        evTimes(find(diff(evTimes)<(12/1000))+1) = [];
                end

            case {'photoDThorLabs','photoDRaw'}
                switch mode
                    case 'default'
                        % Uses k-means here to get thresholds.
                        % Different from the shmitt times I used to use; also takes much longer!
                        % pdT = schmittTimes(timelineTime,chan, [2.5 3]);
                        
                        % Median filter to remove short glitches
                        chan = medfilt1(chan,3);
                        
                        %Kmeans to get thresh. Remove clusters with less
                        %than 2% of points.
                        [clustIdx, thresh] = kmeans(chan(1:5:end),5);
                        thresh(arrayfun(@(x) mean(clustIdx==x)*100, unique(clustIdx))<2) = [];
                        thresh = [min(thresh) + range(thresh)*0.2;  max(thresh) - range(thresh)*0.2];
                        
                        % Find flips based on these thresholds.
                        photoDiodeFlipOn = sort([strfind(chan'>thresh(1), [0 1]), strfind(chan'>thresh(2), [0 1])]);
                        photoDiodeFlipOff = sort([strfind(chan'<thresh(1), [0 1]), strfind(chan'<thresh(2), [0 1])]);
                        photoDiodeFlips = sort([photoDiodeFlipOn photoDiodeFlipOff]);
                        
                        % Remove cases where two flips in the same direction appear in succession (you can't flip to white twice in a row)
                        photoDiodeFlips([strfind(ismember(photoDiodeFlips, photoDiodeFlipOn), [1 1])+1 strfind(ismember(photoDiodeFlips, photoDiodeFlipOff), [1 1])+1]) = [];
                        
                        % Get corresponding flip times. Remove any that would be faster than 60Hz (screen refresh rate)
                        evTimes = timelineTime(photoDiodeFlips)'; % in timeline time
                        evTimes(find(diff(evTimes)<(12/1000))+1) = [];
                    case 'errorMode'
                        % Call this one when the default doesn't work...
                        chan = medfilt1(chan,3);
                        
                        % Find flips based on these thresholds.
                        dchan = diff(chan);
                        [~,photoDiodeFlips] = findpeaks(abs(dchan),'MinPeakProminence',0.2);
                        photoDiodeFlips = photoDiodeFlips';
                        photoDiodeFlips([strfind(dchan(photoDiodeFlips)'>0, [1 1]) ...
                            strfind(dchan(photoDiodeFlips)'<0, [1 1])+1]) = [];
                        
                        % Get corresponding flip times. Remove any that would be faster than 60Hz (screen refresh rate)
                        evTimes = timelineTime(photoDiodeFlips)'; % in timeline time
                        evTimes(find(diff(evTimes)<(12/1000))+1) = [];
                end
                                
            case 'audioOut'
                % Get audio Onsets
                
            case 'camSync'
                % Get cam Sync events (onset of dark flash)
                tlSyncThresh = [min(chan)+0.2*range(chan) max(chan)-0.2*range(chan)]; % these seem to work well
                [~, ~, evTimes] = schmittTimes(timelineTime, chan, tlSyncThresh);
                
            case {'faceCamStrobe','eyeCamStrobe','sideCamStrobe'}
                % Get cam strobe events
                tlStrobeThresh = [1 2];
                [~,evTimes,~] = schmittTimes(timelineTime, chan, tlStrobeThresh);
                
            case 'rewardEcho'
                % Get reward events
                thresh = max(chan)/2;
                rewardTrace = chan > thresh;
                evTimes = strfind(rewardTrace', [0 1])+1;
                evTimes = timelineTime(evTimes); % in timeline time
        end
        
    else
        % Channel couldn't be found. Warning already sent. 
        evTimes = [];
    end

    
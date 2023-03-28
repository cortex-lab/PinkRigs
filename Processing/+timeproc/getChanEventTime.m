function evTimes = getChanEventTime(timeline,chanName,mode)
    %% Extracts event times depending on the chan. 
    %
    % Parameters:
    % -------------------
    % timeline: struct
    %   Timeline struct
    % chanName: str
    %   Name of the channel to extract
    % mode (optional): str
    %   Mode to extract events: default vs. errorMode
    %
    % Returns: 
    % -------------------
    % evTimes: vector
    %   Contains the event times
    
    if ~exist('mode','var')
        mode = 'default';
    end
    
    %% Extract channel
    chan = timeproc.extractChan(timeline,chanName,0);
    timelineTime = timeproc.extractChan(timeline,'time');

    % laserOut channel name might have several endings due to LED indices.
    if contains(chanName,'laserOut')
        chanName = 'laserOut'; 
    end

        
    %% Extract events
    if ~isempty(chan)
        switch chanName
            case 'flipper'
                % Get flips of the flipper
                tlFlipThresh = [min(chan)+0.2*range(chan) max(chan)-0.2*range(chan)]; % these seem to work well
                evTimes = schmittTimes(timelineTime, chan, tlFlipThresh); % all flips, both up and down
    
            case {'photoDiode','photoDThorLabs','photoDRaw'}
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

            case 'laserOut'
                % I detect both the beginning and the end of the ramp. 
                %tlSyncThresh = [min(chan)+0.03*range(chan) max(chan)-0.02*range(chan)];

                % or determine from the baseline in which we always pull
                % down ao in LaserController 
                bl = (chan(1:1000));
                tlSyncThresh = mean(bl)+5*std(bl);
                tlSyncThresh = [tlSyncThresh,tlSyncThresh+50*std(bl)];
                % I need to find a more robust way of doing this. 
                [~, laserOnEnd, laserOffEnd] = schmittTimes(timelineTime, chan, tlSyncThresh);
                [~, laserOffStart, laserOnStart] = schmittTimes(flip(timelineTime), flip(chan), tlSyncThresh);
                laserOnStart = sort(laserOnStart); 
                laserOffStart  = sort(laserOffStart);                 
                evTimes = [laserOnStart,laserOnEnd,laserOffStart,laserOffEnd];

                % throw away events that are point processes and most
                % certainly too long
                laserOnPeriod = diff(evTimes');
                laserOnPeriod = laserOnPeriod(2,:);

                evTimes = evTimes(find(laserOnPeriod>0.1),:); 

            case 'micSync'
                micSyncThresh = [min(chan)+0.2*range(chan) max(chan)-0.2*range(chan)]; % these seem to work well
                [~, evTimes, ~] = schmittTimes(timelineTime, chan, micSyncThresh);

                %

        end
        
    else
        % Channel couldn't be found. Warning already sent. 
        evTimes = [];
    end

    
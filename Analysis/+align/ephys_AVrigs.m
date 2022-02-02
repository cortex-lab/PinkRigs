function b = ephys_AVrigs(expPath)
    %%% This function will align the flipper of the ephys data to the
    %%% flipper taken from the timeline.
    %%%
    %%% This code is inspired by the code from kilotrode
    %%% (https://github.com/cortex-lab/kilotrodeRig).
    
    %% Get parameters
    % Parameters for processing (can be inputs in varargin{1})
    ephysPath = []; % for a specific ephys
    
    % This is not ideal
    if ~isempty(varargin)
        params = varargin{1};
        
        if isfield(params, 'alignType')
            ephysPath = params.ephysPath;
        end
        
        if nargin > 1
            timeline = varargin{2};
        end
    end
    
    %% Get timeline and ephys to loop through
    
    % Get timeline
    if ~exist('timeline','var')
        fprintf(1, 'loading timeline\n');
        timeline = getTimeline(expPath);
    end

    % Get ephys folders
    if isempty(ephysPath)
        [subject, expDate, expNum, server] = parseExpPath(expPath);
        ephysFolder = fullfile(server,'Subjects',subject,expDate,'ephys');
        ephysData = dir(ephysFolder)
        for e = 3:numel(ephysData)
            
        end
    end
    
    
    %% align times (timeline to ephys)
    
    % Detect sync events from timeline
    timelineFlipperTimes = getChanEventTime(timeline,'flipper');

    % Detect sync events for all ephys        

  
    %%  Match up ephys and timeline events
    % Algorithm here is to go through each ephys available, figure out
    % whether the events in timeline align with any of those in the ephys. If
    % so, we have a conversion of events in that ephys into timeline

    if hasEphys
        ef = ephysFlips{1};
        if useFlipper && ef(1)<0.001
            % this happens when the flipper was in the high state to begin with
            % - a turning on event is registered at the first sample. But here
            % we need to drop it.
            ef = ef(2:end);
        end
        for e = 1:length(expNums)
            if hasTimeline(e)
                fprintf('trying to correct timeline %d to ephys\n', expNums(e));
                %Timeline = tl{e};
                tlT = tlFlips{e};
                
                success=false;
                if length(tlT)==length(ef)
                    % easy case: the two are exactly coextensive
                    [~,b] = makeCorrection(ef, tlT, true);
                    success = true;
                elseif length(tlT)<length(ef) && ~isempty(tlT)
                    [~,b,success] = findCorrection(ef, tlT, false);
                elseif length(tlT)>length(ef) && ~isempty(tlT)
                    [~,a,success] = findCorrection(tlT, ef, false);
                    if ~isempty(a)
                        b = [1/a(1); -a(2)/a(1)];
                    end
                end
                if success
                    %                 writeNPY(b, fullfile(alignDir, ...
                    %                     sprintf('correct_timeline_%d_to_ephys_%s.npy', ...
                    %                     e, tags{1})));
                    writeNPY(b, fullfile(alignDir, ...
                        sprintf('correct_timeline_%d_to_ephys_%s.npy', ...
                        expNums(e), tags{1})));
                    fprintf('success\n');
                    eTimeline2keep = e;
                else
                    fprintf('could not correct timeline to ephys\n');
                end
            end
        end
    end
    
    TLexp = expNums(eTimeline2keep);
    Timeline = tl{eTimeline2keep};
    
end
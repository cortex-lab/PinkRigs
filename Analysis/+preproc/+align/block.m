function [blockRefTimes, timelineRefTimes] = block(expPath, varargin)
    %%% This function aligns a block file with its corresponding timeline
    %%% file.
    %%% Additional arguments are some parameters, and the experiment's
    %%% timeline.
    
    %% Get parameters    
    % Parameters for processing (can be inputs in varargin{1})
    params.alignType = [];
    
    % This is not ideal
    if ~isempty(varargin)
        paramsIn = varargin{1};
        params = parseInputParams(params,paramsIn);
        
        if numel(varargin) > 1
            timeline = varargin{2};
        end
    end
    
    %% Get timeline and block
    
    % Get timeline
    if ~exist('timeline','var')
        fprintf(1, 'Loading timeline\n');
        timeline = getTimeline(expPath);
    end
    
    % Get block
    block = getBlock(expPath);
    
    % Get expDef
    % Alignment method will depend on expDef
    expDef = regexp(block.expDef,'\w*\.m','match'); expDef = expDef{1}(1:end-2);
    
    %% Get alignment type

    if ~isempty(params.alignType)
        alignType = params.alignType;
        
        % Check if this specific type would work this would work
        %%% TODO
    else
        % Chose it depending on expDef
        switch expDef
            case 'imageWorld_AllInOne'
                alignType = 'photoDiode';
            case {'multiSpaceWorld'; 'multiSpaceWorld_checker_training'; 'multiSpaceWorld_checker'}
                alignType = 'wheel';
            otherwise 
                fprintf('No alignment type recorded for expDef %s. Using photodiode.\n',expDef)
                alignType = 'photoDiode';
        end
    end
    
    %% Get reference times for block and timeline
    timelineTime = timeline.rawDAQTimestamps;
    sR = 1/diff(timelineTime(1:2));
    switch alignType
        case 'wheel' % Get interpolation points using the wheel data
            % Unwrap the wheel trace (it is circular) and then smooth it. Smoothing is important because covariance will not work otherwise
            smthWin = sR/10+1;
            timelinehWeelPosition = timeproc.extractChan(timeline,'rotaryEncoder');
            timelinehWeelPosition(timelinehWeelPosition > 2^31) = timelinehWeelPosition(timelinehWeelPosition > 2^31) - 2^32;
            timelinehWeelPositionSmooth = smooth(timelinehWeelPosition,smthWin);
            
            % Make initial block time zero (as with timeline) and then interpolate with timeline times and smooth to give same number of points etc.
            block.inputs.wheelValues = block.inputs.wheelValues-block.inputs.wheelValues(1);
            blockWheelPosition = interp1(block.inputs.wheelTimes, block.inputs.wheelValues, timeline.rawDAQTimestamps, 'linear', 'extrap');
            blockWheelPosition = smooth(blockWheelPosition,smthWin);
            
            % Find the overall delay with the entire trace. Then reinterpolate, accounting for this delay.
            baseDelay = finddelay(diff(blockWheelPosition), diff(timelinehWeelPositionSmooth))/sR;
            blockWheelPosition = interp1(block.inputs.wheelTimes+baseDelay, block.inputs.wheelValues, timeline.rawDAQTimestamps, 'linear', 'extrap');
            blockWheelPosition = smooth(blockWheelPosition(:),smthWin);
            
            % Get the vectors for 20s sections of the blockWheelVelocity and timelinehWeelVelocity
            blockWidth = 20*sR;
            sampleCentres = sR*5:sR*10:length(timelineTime);
            blockWheelVelocity = diff(blockWheelPosition);
            timelinehWeelVelocity = diff(timelinehWeelPosition);
            samplePoints = arrayfun(@(x) (x-blockWidth):(x+blockWidth), sampleCentres, 'uni', 0);
            samplePoints = cellfun(@(x) x(x>0 & x<length(timelineTime)), samplePoints, 'uni', 0);
            
            % Check that there is enough wheel movement to make the alignement (based on absolute velocity)
            testIdx = cellfun(@(x) sum(abs(blockWheelVelocity(x))), samplePoints)>(5*blockWidth/sR);
            if mean(testIdx) < 0.2
                error('Not enough movment to synchronize using wheel');
            end
            
            % Go through each subsection and detect the offset between block and timline
            samplePoints = samplePoints(testIdx);
            delayValues = cellfun(@(x) finddelay(blockWheelVelocity(x), timelinehWeelVelocity(x), 1000), samplePoints)./sR;
            
            % Use a smoothed median to select the evolving delat values, and use these to calculate the evolving reference points for block and timeline
            timelineRefTimes = timelineTime(sampleCentres);
            delayValues = interp1(timelineTime(sampleCentres(testIdx)), delayValues, timelineRefTimes, 'linear', 'extrap');
            delayValues = smooth(delayValues, 0.05, 'rlowess');
            blockRefTimes = movmedian(timelineRefTimes(:)-delayValues-baseDelay, 7)';
            blockRefTimes = interp1(timelineRefTimes(4:end-3), blockRefTimes(4:end-3), timelineRefTimes, 'linear', 'extrap');
            block.alignment = 'wheel';
            
        case 'reward' % Get interpolation points using the reward data
            % Rewards are very obvious, but infrequent and sometimes completely absent. But if there is no other option, simply detect the rewards in
            % timesline, and use the rewardTimes from the block outputs to get reference points
            
            % Timeline Ref times
            timelineRefTimes = timeproc.getChanEventTime(timeline,'rewardEcho');
            
            % Block Ref times
            blockRefTimes = block.outputs.rewardTimes(block.outputs.rewardValues > 0);
            
            if length(timelineRefTimes)>length(blockRefTimes); timelineRefTimes = timelineRefTimes(2:end); end
            block.alignment = 'reward';
            
        case 'photoDiode' % Get interpolation points using the photodiode data
            % Note, the photodiode *should* vary between two values, but it often doesn't. For reasons we don't understand, it sometimes goes to grey, and
            % sometimes you skip changes that are recorded in the block etc. This is another reason I prefer to use the wheel. But this method has proved
            % reasonably reliable if the wheel isn't suitable.
            
            % Timeline Ref times
            % Extract photodiode trace and get repeated values by using kmeans. Get the lower and upper thersholds from this range.
            timelineRefTimes = timeproc.getChanEventTime(timeline,'photoDiode');
            % tlRefTimes = tlRefTimes(diff(tlRefTimes)>0.49); % Not sure why is used to be tlRefTimes(diff(tlRefTimes)>0.49). Ask Pip.
            
            % Block Ref times
            blockRefTimes = block.stimWindowUpdateTimes;
            % blRefTimes = block.stimWindowUpdateTimes(diff(block.stimWindowUpdateTimes)>0.49);
            
            % Use "prc.try2alignVectors" to deal with cases where the timeline and block flip times are different lengths, or have large differences. I
            % have found this to solve all problems like this. However, I have also found it to be critical (the photodiode is just too messy otherwise)
            if length(blockRefTimes) ~= length(timelineRefTimes)
                [timelineRefTimes, blockRefTimes] = try2alignVectors(timelineRefTimes, blockRefTimes, 0.25,0);
            elseif any(abs((blockRefTimes-blockRefTimes(1)) - (timelineRefTimes-timelineRefTimes(1)))>0.5)
                [timelineRefTimes, blockRefTimes] = try2alignVectors(timelineRefTimes, blockRefTimes, 0.25,0);
            end
            block.alignment = 'photodiode';
            if length(blockRefTimes) ~= length(timelineRefTimes)
                error('Photodiode alignment error');
            end
    end
    
    %% saves it somewhere?

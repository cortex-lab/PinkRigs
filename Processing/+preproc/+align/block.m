function [blockRefTimes, timelineRefTimes] = block(varargin)
    %% Aligns a block file with its corresponding timeline file.
    % Note: only works for sessions beyond Nov 2022
    %
    % Parameters:
    % -------------------
    % Classic PinkRigs inputs (optional).
    % alignType: str
    %   What to use for alignment. Can be:
    %       'photoDiode', 'wheel'
    % timeline: struct
    %   Corresponding timeline structure.
    %
    % Returns: 
    % -------------------
    % blockRefTimes: vector
    %   Extracted reference times from the block file.
    % timelineRefTimes: vector
    %   Extracted reference times from timeline.

    %% Get parameters
    % Parameters for processing (can be inputs in varargin{1})
    varargin = ['alignType', {[]}, varargin]; % for specific ephys folders (give full path)
    varargin = ['timeline', {[]}, varargin]; % for timeline input
    params = csv.inputValidation(varargin{:});

    %% Get timeline and block
    if ~isfield(params,'dataTimeline')
        fprintf(1, 'Loading timeline\n');
        loadedData = csv.loadData(params, 'dataType', 'timelineblock');
        timeline = loadedData.dataTimeline{1};
        block = loadedData.dataBlock{1};
    else
        loadedData = csv.loadData(params, 'dataType', 'block');
        timeline = params.dataTimeline{1};
        block = loadedData.dataBlock{1};
    end

    %% Get alignment type

    if ~isempty(params.alignType{1})
        alignType = params.alignType{1};

        % Check if this specific type would work this would work
        %%% TODO
    else
        % Chose it depending on expDef
        switch params.expDef{1}{1}
            case {'imageWorld_AllInOne'; 'AVPassive_ckeckerboard_postactive';...
                    'AP_sparseNoise';'AVPassive_checkerboard_extended'}
                alignType = 'photoDiode';
            case {'multiSpaceWorld'; 'multiSpaceWorld_checker_training'; 'multiSpaceWorld_checker'; 'multiSpaceSwitchWorld'}
                alignType = 'wheel';
            case {'spontaneousActivity'}
                alignType = 'none';
                blockRefTimes = nan;
                timelineRefTimes = nan;
            otherwise
                %deals with Tim's vid files for now...
                if contains(params.expDef{1}{1}, 'Vid')
                    alignType = 'none';
                    blockRefTimes = nan;
                    timelineRefTimes = nan;
                else
                    fprintf('No alignment type recorded for expDef %s. Using photodiode.\n',params.expDef{1}{1})
                    alignType = 'photoDiode';
                end
        end
    end

    %% Get reference times for block and timeline
    timelineTime = timeline.rawDAQTimestamps;
    sR = 1/diff(timelineTime(1:2));
    if strcmpi(alignType, 'wheel')
        % Get interpolation points using the wheel data
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
            warning('Not enough movment to synchronize using wheel');
            alignType = 'photodiode';
        else
            % Go through each subsection and detect the offset between block and timline
            samplePoints = samplePoints(testIdx);
            delayValues = cellfun(@(x) finddelay(blockWheelVelocity(x), timelinehWeelVelocity(x), 1000), samplePoints)./sR;

            % Use a smoothed median to select the evolving delay values, and use these to calculate the evolving reference points for block and timeline
            timelineRefTimes = timelineTime(sampleCentres);
            delayValues = interp1(timelineTime(sampleCentres(testIdx)), delayValues, timelineRefTimes, 'linear', 'extrap');
            delayValues = smooth(delayValues, 0.05, 'rlowess');
            blockRefTimes = movmedian(timelineRefTimes(:)-delayValues-baseDelay, 7)';
            blockRefTimes = interp1(timelineRefTimes(4:end-3), blockRefTimes(4:end-3), timelineRefTimes, 'linear', 'extrap');
            block.alignment = 'wheel';
        end
    end

    if strcmpi(alignType, 'photodiode')
        % Get interpolation points using the photodiode data
        % Note, the photodiode *should* vary between two values, but it often doesn't. For reasons we don't understand, it sometimes goes to grey, and
        % sometimes you skip changes that are recorded in the block etc. This is another reason I prefer to use the wheel. But this method has proved
        % reasonably reliable if the wheel isn't suitable.

        % Block Ref times
        blockRefTimes = block.stimWindowUpdateTimes;


        % Timeline Ref times
        % Extract photodiode trace and get repeated values by using kmeans. Get the lower and upper thersholds from this range.
        [timelineRefTimes, photoName] = timeproc.extractBestPhotodiode(timeline, block);
        
        % Use "prc.try2alignVectors" to deal with cases where the timeline and block flip times are different lengths, or have large differences. I
        % have found this to solve all problems like this. However, I have also found it to be critical (the photodiode is just too messy otherwise)
        if length(blockRefTimes) ~= length(timelineRefTimes)
            try
                [timelineRefTimes, blockRefTimes] = try2alignVectors(timelineRefTimes, blockRefTimes);
            catch
                warning('Passing in error mode to get photodiode flip times')
                timelineRefTimes = timeproc.getChanEventTime(timeline, photoName,'errorMode');
                [timelineRefTimes, blockRefTimes] = try2alignVectors(timelineRefTimes, blockRefTimes, 0.25,0);
            end
        elseif any(abs((blockRefTimes-blockRefTimes(1)) - (timelineRefTimes-timelineRefTimes(1)))>0.5)
            [timelineRefTimes, blockRefTimes] = try2alignVectors(timelineRefTimes, blockRefTimes, 0.25,0);
        end
        block.alignment = 'photodiode';
        if length(blockRefTimes) ~= length(timelineRefTimes)
            error('Photodiode alignment error');
        end
    end

    if strcmpi(alignType, 'reward')
        % Get interpolation points using the reward data
        % Rewards are very obvious, but infrequent and sometimes completely absent. But if there is no other option, simply detect the rewards in
        % timesline, and use the rewardTimes from the block outputs to get reference points

        % Timeline Ref times
        timelineRefTimes = timeproc.getChanEventTime(timeline,'rewardEcho');

        % Block Ref times
        blockRefTimes = block.outputs.rewardTimes(block.outputs.rewardValues > 0);

        if length(timelineRefTimes)>length(blockRefTimes); timelineRefTimes = timelineRefTimes(2:end); end
        block.alignment = 'reward';
    end
end
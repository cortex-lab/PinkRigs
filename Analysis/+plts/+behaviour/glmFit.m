function glmData = glmFit(varargin)
%% Generates GLM plots for the behaviour of a mouse/mice
% 
% NOTE: This function uses csv.inputValidate to parse inputs. Paramters are 
% name-value pairs, including those specific to this function
% 
% Parameters: 
% ---------------
% Classic PinkRigs inputs (optional)
%
% modelString (default={'simpLogSplitVSplitA'}): string 
%   Indicates which model to fit. Different strings are interpreted in
%   the script "GLMMultiModels.m" 
%
% plotCond (default={'none'}): string 
%   Plotting with a condition, e.g. on the choice in the previous trial.
%
% sepPlots (default=0): int 
%   If this is a 1, indicates that plots for a single mouse should be shown
%   separately across sessions (rather than combining into an average).
%   
% expDef (default='t'): string
%   String indicating which experiment types to include (see
%   csv.inputValidation, but this will usually be "t" indicating
%   behavioural sessions
% 
% plotType (default='res'): string 
%   If 'log' then log(probR/ProbL) will be plotted on the y-axis
%
% noPlot (default={0}): logical 
%   Indicates whether the actual plotting should be skipped (retuning just data)
%
% contrastPower (default={0}): double 
%   If you want to use a specific contrast power for the 'log' plot. If
%   this is zero, then power calculated in fitting is usedd for each plot
%
% cvFolds (default={0}): int 
%   Indicates the number of cross-validation folds to use
%
% useCurrentAxes (default={0}): logical
%   If 1, will use the current axes to make the plot, rather than
%   generating new axes/figure
%
% onlyPlt (default={0}): logical 
%   If 1, will plot data without actually fitting. 
%   NOTE: this is only for use when plotting GLM data already "extracted"
% 
% fitLineStyle (default={'-'}): str
%     linestype of fitted plot 
%  'datDotStyle' (default={'.'}): str
%     the dot style of the plot use 
% 'useLaserTrials' (default={0}) : double 
%     whether to use the laster trials for glmfit
%  'laserTrialType' (default={1}: double 
%     -1 - left hemisphere inhibtion
%     1 - right hemisphere inhibition (most of the time, depending on LED wiring, recorded in metadata...)

%
% Returns: 
% -----------
% glmData: cell array. One cell per plot, contains GLMmulti class object 
%   .modelString: modelString used
%   .plotCond:    condition for plotting
%   .prmLabels:   labels for the parameters used
%   .prmFits:     fitted values for paramters used
%   .prmBounds:   bounds used for fitting (not confidence interval)
%   .prmInit:     initial values for paramters used
%   .dataBlock:   behaviour data used for fitting (struct)
%   .pHat:        fitting information
%   .logLik:      logliklihood for final fit
%   .evalPoints:  points at which the curve was evaluated
%   .initGuess:   inital guess for values
%
% Examples: 
% ------------
% glmData = plts.behaviour.glmFit('subject', {'AV009'}, 'expDate', 'last5', 'sepPlots', 1)
% glmData = plts.behaviour.glmFit('subject', {'AV009'}, 'expDate', 'last5', 'modelString', 'visOnly')
% glmData = plts.behaviour.glmFit('subject', {'AV008'; 'AV009'}, 'expDate', 'last5', 'plotType', 'log')
% glmData = plts.behaviour.glmFit('subject', {'AV008'; 'AV009'}, 'expDate', 'last5', 'plotType', 'log')
% expList = csv.queryExp(subject='AV008',expDate='last5',expDef='t',sepPlots=1); plts.behaviour.glmFit(expList)


varargin = ['modelString', {{'simpLogSplitVSplitA'}}, varargin];
varargin = ['plotCond', {'none'}, varargin];
varargin = ['sepPlots', {0}, varargin];
varargin = ['expDef', {'t'}, varargin];
varargin = ['plotType', {'res'}, varargin];
varargin = ['noPlot', {0}, varargin];
varargin = ['contrastPower', {0}, varargin];
varargin = ['cvFolds', {0}, varargin];
varargin = ['useCurrentAxes', {0}, varargin];
varargin = ['onlyPlt', {0}, varargin];
varargin = ['fitLineStyle', {'-'}, varargin];
varargin = ['datDotStyle', {'.'}, varargin];
varargin = ['useLaserTrials', {0}, varargin];
varargin = ['laserTrialType', {1}, varargin];


% Deals with sepPlots=1 where subjects are repliacted in getTrainingData
params = csv.inputValidation(varargin{:});
extracted = plts.behaviour.getTrainingData(varargin{:});
if ~any(extracted.validSubjects)
    fprintf('WARNING: No data found for requested subjects... Returning \n');
    return
end
axesOpt.totalNumOfAxes = sum(extracted.validSubjects);
axesOpt.btlrMargins = [80 100 80 40];
axesOpt.gapBetweenAxes = [100 60];
axesOpt.numOfRows = max([1 ceil(axesOpt.totalNumOfAxes/4)]);
axesOpt.figureHWRatio = 1.1;

glmData = cell(length(extracted.data), numel(params.modelString{1}));

for mm = 1:numel(params.modelString{1})
    if ~params.noPlot{1} && ~params.useCurrentAxes{1}; figure('Name',params.modelString{1}{mm}); end
    for i = find(extracted.validSubjects)'
        refIdx = min([i length(params.useCurrentAxes)]);
        if ~params.onlyPlt{refIdx}
            currBlock = extracted.data{i};

            % Add previous choices and rewards
            currBlock.previous_respDirection = [0; currBlock.response_direction(1:end-1)];
            currBlock.previous_respFeedback = [0; currBlock.response_feedback(1:end-1)];

            if params.useLaserTrials{1} && isnan(params.laserTrialType{1})
                keepIdx = currBlock.response_direction & currBlock.is_validTrial & currBlock.is_laserTrial & abs(currBlock.stim_audAzimuth)~=30;
            elseif params.useLaserTrials{1} && ~isnan(params.laserTrialType{1})
                keepIdx = currBlock.response_direction & currBlock.is_validTrial & currBlock.is_laserTrial & abs(currBlock.stim_audAzimuth)~=30 & currBlock.stim_laserPosition==params.laserTrialType{1};
            elseif (params.useLaserTrials{1}==0) && sum(isnan(currBlock.is_laserTrial))==0
                keepIdx = currBlock.response_direction & currBlock.is_validTrial & ~currBlock.is_laserTrial & abs(currBlock.stim_audAzimuth)~=30;
            else
                keepIdx = currBlock.response_direction & currBlock.is_validTrial & abs(currBlock.stim_audAzimuth)~=30;

            end
            currBlock = filterStructRows(currBlock, keepIdx);
            glmData{i,mm} = plts.behaviour.GLMmulti(currBlock, params.modelString{refIdx}{mm});
        else
            glmData{i,mm} = extracted.data{i};
        end

        if params.useCurrentAxes{refIdx}; obj.hand.axes = gca;
        elseif ~params.noPlot{refIdx}; obj.hand.axes = plts.general.getAxes(axesOpt, i);
        end

        if ~params.onlyPlt{refIdx}
            if ~params.cvFolds{refIdx}; glmData{i,mm}.fit; end
            if params.cvFolds{refIdx}; glmData{i,mm}.fitCV(params.cvFolds{refIdx}); end
        end
        if params.noPlot{1}; continue; end

        %%% FIT
        params2use = mean(glmData{i,mm}.prmFits,1);
        pHatCalculated = glmData{i,mm}.calculatepHat(params2use,'eval');

        visValFit = glmData{i,mm}.evalPoints(:,1);
        audValFit = glmData{i,mm}.evalPoints(:,2);
        if size(glmData{i,mm}.evalPoints,2)>3
            prevRespDirValFit = glmData{i,mm}.evalPoints(:,3);
            prevRespDirValFit(prevRespDirValFit == 1) = -2; % -2 = left, 0 = timeout, 2 = right
            prevRespDirValFit = prevRespDirValFit/2; % between -1 and 1
            prevRespFeedValFit = glmData{i,mm}.evalPoints(:,4);
        end

        clear uniGridFit
        uniGridFit{1} = unique(visValFit);
        uniGridFit{2} = unique(audValFit);
        allVal = [visValFit,audValFit];
        switch params.modelString{refIdx}{mm}
            case 'simpLogSplitVSplitAPast'
                % Split by previous choice
                uniGridFit = cat(2,uniGridFit,{unique(prevRespDirValFit)});
                allVal = cat(2,allVal,prevRespDirValFit);
                noRepIdx = 1:numel(uniGridFit{1})*numel(uniGridFit{2})*numel(uniGridFit{3});
                allVal = allVal(noRepIdx,:);
                pHatCalculated = pHatCalculated(noRepIdx,:);

                lineStyle = {params.fitLineStyle{1}; params.fitLineStyle{1}; params.fitLineStyle{1}};
                lineColors_tmp = plts.general.selectRedBlueColors(uniGridFit{2});
                lineColors = {repmat(lineColors_tmp(1,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(2,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(3,:),[numel(uniGridFit{2}) 1])};

            case {'simpLogSplitVSplitAPastWSLS', 'simpLogSplitVSplitAPastBusse'}
                % Split by previous choice and previous reward
                prevRewardedChoice = prevRespDirValFit;
                prevRewardedChoice(prevRespDirValFit == -1 & prevRespFeedValFit == 1) = -2;
                prevRewardedChoice(prevRespDirValFit ==1 & prevRespFeedValFit == 1) = 2;

                uniGridFit = cat(2,uniGridFit,{unique(prevRewardedChoice)});
                allVal = cat(2,allVal,prevRewardedChoice);

                lineStyle = {'-','--',':','--','-'};
                lineColors_tmp = plts.general.selectRedBlueColors(uniGridFit{2});
                lineColors = {repmat(lineColors_tmp(1,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(1,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(2,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(3,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(3,:),[numel(uniGridFit{2}) 1])};

            otherwise
                % Nothing to add!

                lineStyle = {params.fitLineStyle{1}};
                lineColors = {plts.general.selectRedBlueColors(uniGridFit{2})};
        end
        clear valGridFit
        [valGridFit{1:numel(uniGridFit)}] = ndgrid(uniGridFit{:});

        [~, gridIdx] = ismember(allVal, cell2mat(cellfun(@(x) x(:), valGridFit, 'uni', 0)), 'rows');
        plotData = valGridFit{1};
        plotData(gridIdx) = pHatCalculated(:,2);
        plotOpt.lineStyle = params.fitLineStyle{1};
        plotOpt.Marker = 'none';

        if strcmp(params.plotType{refIdx}, 'log')
            if ~params.contrastPower{refIdx}
                params.contrastPower{refIdx}  = params2use(strcmp(glmData{i,mm}.prmLabels, 'N'));
            end
            if isempty(params.contrastPower{refIdx})
                tempFit = plts.behaviour.GLMmulti(currBlock, 'simpLogSplitVSplitA');
                tempFit.fit;
                tempParams = mean(tempFit.prmFits,1);
                params.contrastPower{refIdx}  = tempParams(strcmp(tempFit.prmLabels, 'N'));
            end
            plotData = log10(plotData./(1-plotData));
        else
            params.contrastPower{refIdx} = 1;
        end
        contrastPower = params.contrastPower{refIdx};
        visValues = (abs(valGridFit{1}(:,1))).^contrastPower.*sign(valGridFit{1}(:,1));

        for k = 1:size(plotData,3)
            plotOpt.lineStyle = lineStyle{k};
            plts.general.rowsOfGrid(visValues, plotData(:,:,k)', lineColors{k}, plotOpt);
        end

        %%% DATA
        visDiff = currBlock.stim_visDiff;
        audDiff = currBlock.stim_audDiff;
        responseDir = currBlock.response_direction;
        prevRespDir = currBlock.previous_respDirection;
        prevRespDir(prevRespDir == 1) = -2; % -2 = left, 0 = timeout, 2 = right
        prevRespDir = prevRespDir/2; % between -1 and 1
        prevRespFeed = currBlock.previous_respFeedback;

        % Get trial conditions in a grid
        clear uniGrid
        uniGrid{1} = unique(visDiff);
        uniGrid{2} = unique(audDiff);
        blkSumm = [visDiff,audDiff];
        switch params.plotCond{refIdx}
            case 'none'
                % Nothing to add!
                Marker = {params.datDotStyle(1)};
                lineColors = {plts.general.selectRedBlueColors(uniGrid{2})};

            case 'prevChoice'
                uniGrid = cat(2,uniGrid,{unique(prevRespDir)});
                blkSumm = cat(2,blkSumm,prevRespDir);

                plotOpt.MarkerSize = 10;
                Marker = {'o','o','o'};
                lineColors_tmp = plts.general.selectRedBlueColors(uniGridFit{2});
                lineColors = {repmat(lineColors_tmp(1,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(2,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(3,:),[numel(uniGridFit{2}) 1])};

            case 'prevRewardedChoice'
                % Right of Left, rewarded or not
                prevRewardedChoice = prevRespDir;
                prevRewardedChoice(prevRespDir == -1 & prevRespFeed == 1) = -2;
                prevRewardedChoice(prevRespDir == 1 & prevRespFeed == 1) = 2;
                uniGrid = cat(2,uniGrid,{unique(prevRewardedChoice)});
                blkSumm = cat(2,blkSumm,prevRewardedChoice);

                plotOpt.MarkerSize = 10;
                Marker = {'o','x','.','x','o'};
                lineColors_tmp = plts.general.selectRedBlueColors(uniGridFit{2});
                lineColors = {repmat(lineColors_tmp(1,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(1,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(2,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(3,:),[numel(uniGridFit{2}) 1]); ...
                    repmat(lineColors_tmp(3,:),[numel(uniGridFit{2}) 1])};
        end
        clear valGrid
        [valGrid{1:numel(uniGrid)}] = ndgrid(uniGrid{:});
        maxContrast = max(abs(visDiff));

        % Find fraction of right turn for each condition -- less elegant than
        % before but couldn't find a shorter way
        fracRightTurns = nan(1,numel(valGrid{1}(:)));
        for k = 1:numel(valGrid{1}(:))
            idx = [];
            for c = 1:size(blkSumm,2)
                idx = cat(2,idx,ismember(blkSumm(:,c),valGrid{c}(k)));
            end
            idx = all(idx,2);
            fracRightTurns(k) = mean(responseDir(idx)==2);
        end
        fracRightTurns = reshape(fracRightTurns,size(valGrid{1}));

        visValues = abs(uniGrid{1}).^contrastPower.*sign(uniGrid{1})./(maxContrast.^contrastPower);
        if strcmp(params.plotType{refIdx}, 'log')
            fracRightTurns = log10(fracRightTurns./(1-fracRightTurns));
        end
        plotOpt.lineStyle = 'none';
        s = size(valGrid{1});
        for k = 1:prod(s(3:end))
            plotOpt.Marker = Marker{k};
            plts.general.rowsOfGrid(visValues, fracRightTurns(:,:,k)', lineColors{k}, plotOpt);
        end

        xlim([-1 1])
        midPoint = 0.5;
        xTickLoc = (-1):(1/8):1;
        if strcmp(params.plotType{refIdx}, 'log')
            ylim([-2.6 2.6])
            midPoint = 0;
            xTickLoc = sign(xTickLoc).*abs(xTickLoc).^contrastPower;
        end

        box off;
        xTickLabel = num2cell(round(((-maxContrast):(maxContrast/8):maxContrast)*100));
        xTickLabel(2:2:end) = deal({[]});
        set(gca, 'xTick', xTickLoc, 'xTickLabel', xTickLabel);

        if params.useLaserTrials{1}
            if unique(currBlock.stim_laserPosition)==-1; inhibited_hemisphere = 'Left'; elseif unique(currBlock.stim_laserPosition)==1; inhibited_hemisphere = 'Right'; end
            titlestring = sprintf('%s: %d opto Tri in %s,%s hemi', extracted.subject{i}, length(responseDir), extracted.blkDates{i}{1},inhibited_hemisphere);

        else
            titlestring = sprintf('%s: %d Tri in %s', extracted.subject{i}, length(responseDir), extracted.blkDates{i}{1});
        end

        title(titlestring)
        xL = xlim; hold on; plot(xL,[midPoint midPoint], '--k', 'linewidth', 1.5);
        yL = ylim; hold on; plot([0 0], yL, '--k', 'linewidth', 1.5);
    end
end
end
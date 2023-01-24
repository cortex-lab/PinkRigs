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
% Returns: 
% -----------
% glmData: cell array. One cell per plot, contains GLMmulti class object 
%   .modelString: modelString used
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
% glmData = plt.behaviour.glmFit('subject', {'AV009'}, 'expDate', 'last5', 'sepPlots', 1)
% glmData = plt.behaviour.glmFit('subject', {'AV009'}, 'expDate', 'last5', 'modelString', 'visOnly')
% glmData = plt.behaviour.glmFit('subject', {'AV008'; 'AV009'}, 'expDate', 'last5', 'plotType', 'log')
% glmData = plt.behaviour.glmFit('subject', {'AV008'; 'AV009'}, 'expDate', 'last5', 'plotType', 'log')
% expList = csv.queryExp(subject='AV008',expDate='last5',expDef='t',sepPlots=1); plt.behaviour.glmFit(expList)


varargin = ['modelString', {'simpLogSplitVSplitA'}, varargin];
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



% Deals with sepPlots=1 where subjects are repliacted in getTrainingData
params = csv.inputValidation(varargin{:});
extracted = plt.behaviour.getTrainingData(varargin{:});
if ~any(extracted.validSubjects)
    fprintf('WARNING: No data found for requested subjects... Returning \n');
    return
end
axesOpt.totalNumOfAxes = sum(extracted.validSubjects);
axesOpt.btlrMargins = [80 100 80 40];
axesOpt.gapBetweenAxes = [100 60];
axesOpt.numOfRows = max([1 ceil(axesOpt.totalNumOfAxes/4)]);
axesOpt.figureHWRatio = 1.1;

glmData = cell(length(extracted.data), 1);
if ~params.noPlot{1} && ~params.useCurrentAxes{1}; figure; end
for i = find(extracted.validSubjects)'
    refIdx = min([i length(params.useCurrentAxes)]);
    if ~params.onlyPlt{refIdx}
        currBlock = extracted.data{i};

        if params.useLaserTrials{1}
            keepIdx = currBlock.response_direction & currBlock.is_validTrial & currBlock.is_laserTrial & abs(currBlock.stim_audAzimuth)~=30;
        elseif (params.useLaserTrials{1}==0) && sum(isnan(currBlock.is_laserTrial))==0
            keepIdx = currBlock.response_direction & currBlock.is_validTrial & ~currBlock.is_laserTrial & abs(currBlock.stim_audAzimuth)~=30;
        else
            keepIdx = currBlock.response_direction & currBlock.is_validTrial & abs(currBlock.stim_audAzimuth)~=30;

        end
        currBlock = filterStructRows(currBlock, keepIdx);
        glmData{i} = plt.behaviour.GLMmulti(currBlock, params.modelString{refIdx});
    else
        glmData{i} = extracted.data{i};
    end

    if params.useCurrentAxes{refIdx}; obj.hand.axes = gca; 
    elseif ~params.noPlot{refIdx}; obj.hand.axes = plt.general.getAxes(axesOpt, i); 
    end
    
    if ~params.onlyPlt{refIdx}
        if ~params.cvFolds{refIdx}; glmData{i}.fit; end
        if params.cvFolds{refIdx}; glmData{i}.fitCV(params.cvFolds{refIdx}); end
    end
    if params.noPlot{1}; return; end
    
    params2use = mean(glmData{i}.prmFits,1);   
    pHatCalculated = glmData{i}.calculatepHat(params2use,'eval');
    [grids.visValues, grids.audValues] = meshgrid(unique(glmData{i}.evalPoints(:,1)),unique(glmData{i}.evalPoints(:,2)));
    [~, gridIdx] = ismember(glmData{i}.evalPoints, [grids.visValues(:), grids.audValues(:)], 'rows');
    plotData = grids.visValues;
    plotData(gridIdx) = pHatCalculated(:,2);
    plotOpt.lineStyle = params.fitLineStyle{1};
    plotOpt.Marker = 'none';

    if strcmp(params.plotType{refIdx}, 'log')
        if ~params.contrastPower{refIdx}
            params.contrastPower{refIdx}  = params2use(strcmp(glmData{i}.prmLabels, 'N'));
        end
        if isempty(params.contrastPower{refIdx})
            tempFit = plt.behaviour.GLMmulti(currBlock, 'simpLogSplitVSplitA');
            tempFit.fit;
            tempParams = mean(tempFit.prmFits,1);
            params.contrastPower{refIdx}  = tempParams(strcmp(tempFit.prmLabels, 'N'));
        end
        plotData = log10(plotData./(1-plotData));
    else
        params.contrastPower{refIdx} = 1;
    end
    contrastPower = params.contrastPower{refIdx};

    visValues = (abs(grids.visValues(1,:))).^contrastPower.*sign(grids.visValues(1,:));
    lineColors = plt.general.selectRedBlueColors(grids.audValues(:,1));
    plt.general.rowsOfGrid(visValues, plotData, lineColors, plotOpt);

    plotOpt.lineStyle = 'none';
    plotOpt.Marker = params.datDotStyle{1};
    
    visDiff = currBlock.stim_visDiff;
    audDiff = currBlock.stim_audDiff;
    responseDir = currBlock.response_direction;
    [visGrid, audGrid] = meshgrid(unique(visDiff),unique(audDiff));
    maxContrast = max(abs(visGrid(1,:)));
    fracRightTurns = arrayfun(@(x,y) mean(responseDir(ismember([visDiff,audDiff],[x,y],'rows'))==2), visGrid, audGrid);
    
    visValues = abs(visGrid(1,:)).^contrastPower.*sign(visGrid(1,:))./(maxContrast.^contrastPower);
    if strcmp(params.plotType{refIdx}, 'log')
        fracRightTurns = log10(fracRightTurns./(1-fracRightTurns));
    end
    plt.general.rowsOfGrid(visValues, fracRightTurns, lineColors, plotOpt);
    
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

    title(sprintf('%s: %d Tri in %s', extracted.subject{i}, length(responseDir), extracted.blkDates{i}{1}))
    xL = xlim; hold on; plot(xL,[midPoint midPoint], '--k', 'linewidth', 1.5);
    yL = ylim; hold on; plot([0 0], yL, '--k', 'linewidth', 1.5);
end
end
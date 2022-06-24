function glmData = glmFit(varargin)
%% Generate GLM plots for the behaviour of a mouse/mice
%% Input validation and default assingment
varargin = ['modelString', {'simpLogSplitVSplitA'}, varargin];
varargin = ['cvFolds', {0}, varargin];
varargin = ['contrastPower', {0}, varargin];
varargin = ['sepPlots', {0}, varargin];
varargin = ['expDef', {'t'}, varargin];
varargin = ['plotType', {'res'}, varargin];
varargin = ['noPlot', {0}, varargin];
varargin = ['onlyPlt', {0}, varargin];
varargin = ['useCurrentAxes', {0}, varargin];

extracted = plt.behaviour.getTrainingData(varargin{:});
params = csv.inputValidation(varargin{:});

axesOpt.totalNumOfAxes = sum(extracted.validSubjects);
axesOpt.btlrMargins = [80 100 80 40];
axesOpt.gapBetweenAxes = [100 60];
axesOpt.numOfRows = max([2 ceil(axesOpt.totalNumOfAxes/4)]);
axesOpt.figureHWRatio = 1.1;

glmData = cell(length(extracted.data), 1);
if ~params.noPlot{1} && ~params.useCurrentAxes{1}; figure; end
for i = find(extracted.validSubjects)'
    if ~params.onlyPlt{1}
        currBlock = extracted.data{i};
        keepIdx = currBlock.response_direction & currBlock.is_validTrial;
        currBlock = filterStructRows(currBlock, keepIdx);
        glmData{i} = plt.behaviour.GLMmulti(currBlock, params.modelString{i});
    else
        glmData{i} = extracted.data{i};
    end

    if params.useCurrentAxes{i}; obj.hand.axes = gca; 
    elseif ~params.noPlot{i}; obj.hand.axes = plt.general.getAxes(axesOpt, i); 
    end
    
    if ~params.onlyPlt{i}
        if ~params.cvFolds{i}; glmData{i}.fit; end
        if params.cvFolds{i}; glmData{i}.fitCV(params.cvFolds{i}); end
    end
    if params.noPlot{1}; return; end
    
    params2use = mean(glmData{i}.prmFits,1);   
    pHatCalculated = glmData{i}.calculatepHat(params2use,'eval');
    [grids.visValues, grids.audValues] = meshgrid(unique(glmData{i}.evalPoints(:,1)),unique(glmData{i}.evalPoints(:,2)));
    [~, gridIdx] = ismember(glmData{i}.evalPoints, [grids.visValues(:), grids.audValues(:)], 'rows');
    plotData = grids.visValues;
    plotData(gridIdx) = pHatCalculated(:,2);
    plotOpt.lineStyle = '-';
    plotOpt.Marker = 'none';

    if strcmp(params.plotType{i}, 'log')
        if ~params.contrastPower{i}
            params.contrastPower{i}  = params2use(strcmp(glmData{i}.prmLabels, 'N'));
        end
        if isempty(params.contrastPower{i})
            tempFit = plt.behaviour.GLMmulti(currBlock, 'simpLogSplitVSplitA');
            tempFit.fit;
            tempParams = mean(tempFit.prmFits,1);
            params.contrastPower{i}  = tempParams(strcmp(tempFit.prmLabels, 'N'));
        end
        plotData = log10(plotData./(1-plotData));
    else
        params.contrastPower{i} = 1;
    end
    contrastPower = params.contrastPower{i};

    visValues = (abs(grids.visValues(1,:))).^contrastPower.*sign(grids.visValues(1,:));
    lineColors = plt.selectRedBlueColors(grids.audValues(:,1));
    plt.rowsOfGrid(visValues, plotData, lineColors, plotOpt);

    plotOpt.lineStyle = 'none';
    plotOpt.Marker = '.';
    
    visDiff = currBlock.stim_visDiff;
    audDiff = currBlock.stim_audDiff;
    responseDir = currBlock.response_direction;
    [visGrid, audGrid] = meshgrid(unique(visDiff),unique(audDiff));
    maxContrast = max(abs(visGrid(1,:)));
    fracRightTurns = arrayfun(@(x,y) mean(responseDir(ismember([visDiff,audDiff],[x,y],'rows'))==2), visGrid, audGrid);
    
    visValues = abs(visGrid(1,:)).^contrastPower.*sign(visGrid(1,:))./(maxContrast.^contrastPower);
    if strcmp(params.plotType{i}, 'log')
        fracRightTurns = log10(fracRightTurns./(1-fracRightTurns));
    end
    plt.rowsOfGrid(visValues, fracRightTurns, lineColors, plotOpt);
    
    xlim([-1 1])
    midPoint = 0.5;
    xTickLoc = (-1):(1/8):1;
    if strcmp(params.plotType{i}, 'log')
        ylim([-2.6 2.6])
        midPoint = 0;
        xTickLoc = sign(xTickLoc).*abs(xTickLoc).^contrastPower;
    end

    box off;
    xTickLabel = num2cell(round(((-maxContrast):(maxContrast/8):maxContrast)*100));
    xTickLabel(2:2:end) = deal({[]});
    set(gca, 'xTick', xTickLoc, 'xTickLabel', xTickLabel);

    title(sprintf('%s: %d Tri in %d Exp', extracted.subject{i}, length(responseDir), extracted.nExp{i}))
    xL = xlim; hold on; plot(xL,[midPoint midPoint], '--k', 'linewidth', 1.5);
    yL = ylim; hold on; plot([0 0], yL, '--k', 'linewidth', 1.5);
end
end
function glmData = glmFit(varargin)
%% Generate GLM plots for the behaviour of a mouse/mice
%% Input validation and default assingment
varargin = ['modelString', {'simpLogSplitVSplitA'}, varargin];
varargin = ['sepPlots', {nan}, varargin];
varargin = ['expDef', {'t'}, varargin];
varargin = ['plotType', {'res'}, varargin];
varargin = ['noPlot', {0}, varargin];
varargin = ['onlyPlt', {0}, varargin];
varargin = ['useCurrentAxes', {0}, varargin];

params = csv.inputValidation(varargin{:});
extracted = plt.behaviour.getTrainingData(params);

blkDates = extracted.blkDates;
rigNames = extracted.rigNames;

axesOpt.totalNumOfAxes = sum(extracted.validSubjects);
axesOpt.btlrMargins = [80 100 80 40];
axesOpt.gapBetweenAxes = [100 60];
axesOpt.numOfRows = min(length(obj.blks), 4);
axesOpt.figureHWRatio = 1.1;

glmData = cell(length(extracted.data), 1);
if ~params.noPlot{1} && ~useCurrentAxes; figure; end
for i = find(extracted.validSubjects)'
    if ~onlyPlt
        normBlk = prc.filtBlock(normBlk,~isinf(normBlk.tri.stim.audInitialAzimuth));
        obj.glmFit{i} = fit.GLMmulti(normBlk, modelString);
    else
        normBlk = obj.blks(i);
    end






    glmPlot.subject = params.subject{i};
    glmPlot.xyLabel = {'VisualContrast'; 'AuditoryAzimuth'};
    glmPlot.axisLimits = [0 1];
    glmPlot.colorMap = plt.general.redBlueMap(64);

    if isempty(extracted.data{i}) || extracted.data{i}.nExperiments == 1 
        glmPlot.extraInf = [blkDates{i}{1} ' on ' rigNames{i}{1}];
    else
        glmPlot.extraInf = num2str([extracted.data{i}.nExperiments 'Sess']);
    end
    if ~isempty(extracted.data{i})
        tDat = rmfield(extracted.data{i}, {'nExperiments', 'AVParams'});
        glmPlot.nExperiments = extracted.data{i}.nExperiments;
    end

    if isempty(extracted.data{i})
        glmPlot.xyValues = {0; 0};
        glmPlot.plotData = nan;
        glmPlot.trialCount = 0;
        glmPlot.totTrials = nan;
        glmPlot.nExperiments = nan;
    elseif strcmpi(params.plotType{1}(1:3), 'res')
        tkIdx = extracted.data{i}.is_validTrial & extracted.data{i}.response_direction;
        tDat = filterStructRows(tDat, tkIdx);

        [~,~,vLabel] = unique(tDat.stim_visDiff);
        [~,~,aLabel] = unique(tDat.stim_audDiff);
        glmPlot.plotData = accumarray([aLabel, vLabel],tDat.response_direction,[],@mean)-1;
        glmPlot.trialCount = accumarray([aLabel, vLabel],~isnan(tDat.response_direction),[],@sum);
        glmPlot.plotData(glmPlot.trialCount==0) = nan;
        glmPlot.totTrials = length(tDat.stim_visDiff);
        glmPlot.xyValues = {unique(tDat.stim_visDiff)*100; unique(tDat.stim_audDiff)};
        colorBar.colorLabel = 'Fraction of right turns';
        colorBar.colorDirection = 'normal';
        colorBar.colorYTick = {'0'; '1'};
    end
    if ~params.noPlot{1}
        plt.general.getAxes(axesOpt, find(find(extracted.validSubjects)'==i));
        makePlot(glmPlot);
    end
    plotData{i,1} = glmPlot;
end
if ~params.noPlot{1}
    currentAxisPotision = get(gca, 'position');
    figureSize = get(gcf, 'position');

    colorBar.handle = colorbar;
    set(colorBar.handle, 'Ticks', get(colorBar.handle, 'Limits'), 'TickLabels', colorBar.colorYTick, 'YDir', colorBar.colorDirection);
    set(gca, 'position', currentAxisPotision);
    colorBar.textHandle = ylabel(colorBar.handle, colorBar.colorLabel);
    set(colorBar.textHandle, 'position', [1 mean(get(colorBar.handle, 'Limits')) 0], 'FontSize', 14)
    set(colorBar.handle, 'position', [1-75/figureSize(3), 0.2, 30/figureSize(3), 0.6])
end
end


function makePlot(glmPlot, addText)
if ~exist('addText', 'var'); addText = 1; end
if ~isfield(glmPlot, 'plotLabels'); glmPlot.plotLabels = glmPlot.plotData; end
if iscell(glmPlot.subject); glmPlot.subject = glmPlot.subject{1}; end
plotData = glmPlot.plotData;
triNum = glmPlot.trialCount;
imAlpha=ones(size(plotData));
imAlpha(isnan(plotData))=0;
imagesc(plotData, 'AlphaData', imAlpha);
caxis(glmPlot.axisLimits);
colormap(glmPlot.colorMap);
daspect([1 1 1]); axis xy;
[xPnts, yPnts] = meshgrid(1:size(plotData,2), 1:size(plotData,1));

tIdx = ~isnan(plotData);
if addText
    txtD = num2cell([xPnts(tIdx), yPnts(tIdx), round(100*plotData(tIdx))/100, triNum(tIdx)],2);
    cellfun(@(x) text(x(1),x(2), {num2str(x(3)), num2str(x(4))}, 'horizontalalignment', 'center'), txtD)
end
if isfield(glmPlot, 'extraInf')
    title(sprintf('%s: %d Tri, %s', glmPlot.subject, glmPlot.totTrials, glmPlot.extraInf))
else, title(sprintf('%s: %d Tri, %d Sess', glmPlot.subject, glmPlot.totTrials, glmPlot.nExperiments));
end
set(gca, 'xTick', 1:size(plotData,2), 'xTickLabel', glmPlot.xyValues{1}, 'fontsize', 14)
set(gca, 'yTick', 1:size(plotData,1), 'yTickLabel', glmPlot.xyValues{2}, 'fontsize', 14, 'TickLength', [0, 0])
box off;
end
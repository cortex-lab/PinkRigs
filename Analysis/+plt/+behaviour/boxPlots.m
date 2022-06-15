function plotData = boxPlots(varargin)
%% Generate box plots for the behaviour of a mouse/mice
%% Input validation and default assingment
varargin = ['sepPlots', {nan}, varargin];
varargin = ['expDef', {'t'}, varargin];
varargin = ['plotType', {'res'}, varargin];
varargin = ['noPlot', {0}, varargin];
params = csv.inputValidation(varargin{:});
extracted = plt.behaviour.getTrainingData(params);

blkDates = extracted.blkDates;
rigNames = extracted.rigNames;

maxGrid = cellfun(@(x) [length(unique(x.stim_audDiff)) length(unique(x.stim_visDiff))], extracted.data, 'uni', 0);
maxGrid = max(cell2mat(maxGrid),2);
axesOpt.figureHWRatio = maxGrid(2)/(1.3*maxGrid(1));
axesOpt.btlrMargins = [100 80 60 100];
axesOpt.gapBetweenAxes = [100 40];
axesOpt.totalNumOfAxes = sum(extracted.validSubjects);

plotData = cell(length(extracted.data), 1);
if ~params.noPlot{1}; figure; end
for i = find(extracted.validSubjects)'
    boxPlot.subject = params.subject{i};
    boxPlot.xyLabel = {'AuditoryAzimuth'; 'VisualContrast'};
    boxPlot.axisLimits = [0 1];
    boxPlot.colorMap = plt.general.redBlueMap(64);

    if isempty(extracted.data{i}) || extracted.nExp{i} == 1 
        boxPlot.extraInf = [blkDates{i}{1} ' on ' rigNames{i}{1}];
    else
        boxPlot.extraInf = num2str([extracted.nExp{i} 'Sess']);
    end
    if ~isempty(extracted.data{i})
        tDat = extracted.data{i};
        boxPlot.nExp = extracted.nExp{i};
    end

    if isempty(extracted.data{i})
        boxPlot.xyValues = {0; 0};
        boxPlot.plotData = nan;
        boxPlot.trialCount = 0;
        boxPlot.totTrials = nan;
        boxPlot.nExp = nan;
    elseif strcmpi(params.plotType{1}(1:3), 'res')
        keepIdx = tDat.is_validTrial & tDat.response_direction;
        tDat = filterStructRows(tDat, keepIdx);

        [~,~,vLabel] = unique(tDat.stim_visDiff);
        [~,~,aLabel] = unique(tDat.stim_audDiff);
        boxPlot.plotData = accumarray([aLabel, vLabel],tDat.response_direction,[],@mean)-1;
        boxPlot.trialCount = accumarray([aLabel, vLabel],~isnan(tDat.response_direction),[],@sum);
        boxPlot.plotData(boxPlot.trialCount==0) = nan;
        boxPlot.totTrials = length(tDat.stim_visDiff);
        boxPlot.xyValues = {unique(tDat.stim_visDiff)*100; unique(tDat.stim_audDiff)};
        colorBar.colorLabel = 'Fraction of right turns';
        colorBar.colorDirection = 'normal';
        colorBar.colorYTick = {'0'; '1'};
    end
    if ~params.noPlot{1}
        plt.general.getAxes(axesOpt, find(find(extracted.validSubjects)'==i));
        makePlot(boxPlot);
    end
    plotData{i,1} = boxPlot;
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


function makePlot(boxPlot, addText)
if ~exist('addText', 'var'); addText = 1; end
if ~isfield(boxPlot, 'plotLabels'); boxPlot.plotLabels = boxPlot.plotData; end
if iscell(boxPlot.subject); boxPlot.subject = boxPlot.subject{1}; end
plotData = boxPlot.plotData;
triNum = boxPlot.trialCount;
imAlpha=ones(size(plotData));
imAlpha(isnan(plotData))=0;
imagesc(plotData, 'AlphaData', imAlpha);
caxis(boxPlot.axisLimits);
colormap(boxPlot.colorMap);
daspect([1 1 1]); axis xy;
[xPnts, yPnts] = meshgrid(1:size(plotData,2), 1:size(plotData,1));

tIdx = ~isnan(plotData);
if addText
    txtD = num2cell([xPnts(tIdx), yPnts(tIdx), round(100*plotData(tIdx))/100, triNum(tIdx)],2);
    cellfun(@(x) text(x(1),x(2), {num2str(x(3)), num2str(x(4))}, 'horizontalalignment', 'center'), txtD)
end
title(sprintf('%s: %d Tri, %s', boxPlot.subject, boxPlot.totTrials, boxPlot.extraInf))

set(gca, 'xTick', 1:size(plotData,2), 'xTickLabel', boxPlot.xyValues{1}, 'fontsize', 14)
set(gca, 'yTick', 1:size(plotData,1), 'yTickLabel', boxPlot.xyValues{2}, 'fontsize', 14, 'TickLength', [0, 0])
box off;
end
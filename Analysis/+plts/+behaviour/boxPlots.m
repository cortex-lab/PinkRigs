function plotData = boxPlots(varargin)
%% Generates box plots for the behaviour of a mouse/mice
% 
% NOTE: This function uses csv.inputValidate to parse inputs. Paramters are 
% name-value pairs, including those specific to this function
%
% Parameters: 
% ---------------
% Classic PinkRigs inputs (optional)
%
% sepPlots (default=1/0 one/multiple subjects requested): int 
%   If this is a 1, indicates that plots for a single mouse should be shown
%   separately across sessions (rather than combining into an average).
%   
% expDef (default='t'): string
%   String indicating which experiment types to include (see
%   csv.inputValidation, but this will usually be "t" indicating
%   behavioural sessions
% 
% plotType (default='res'): string 
%   This allows for different types of plots to be created, but at the 
%   moment, only 'res' (which is the fraction of rightward choices) exits
%
% noPlot (default={0}): logical 
%   Indicates whether the actual plotting should be skipped (and retuning just the data)
%
% Returns: 
% -----------
% plotData: struct. All fields are cell arrays with one cell per plot.
%   .subject:    subject(s) in the plot
%   .xyLabel:    labels for the axes
%   .axisLimits: colorbar axes limits
%   .colorMap:   colormap used (nx3 matrix)
%   .extraInf:   additional info appended to plot title
%   .nExp:       number of experiments
%   .plotData:   tha actual data values that go into the box plot
%   .trialCount: number of trials for each stimulus condition in plot
%   .totTrials:  total number of trials
%   .xyValues:   values for the axes
%
% Examples: 
% ------------
% plotData = plts.behaviour.boxPlots('subject', {'AV009'}, 'expDate', 'last5', 'sepPlots', 0)
% plotData = plts.behaviour.boxPlots('subject', {'AV008'}, 'noPlot', 1, 'expDate', 'last5')
% plotData = plts.behaviour.boxPlots('subject', {'AV008'; 'AV009'}, 'expDate', 'last5')

%% Input validation and default assingment
varargin = ['sepPlots', {nan}, varargin];
varargin = ['expDef', {'t'}, varargin];
varargin = ['plotType', {'res'}, varargin];
varargin = ['noPlot', {0}, varargin];
extracted = plts.behaviour.getTrainingData(varargin{:});
if ~any(extracted.validSubjects)
    fprintf('WARNING: No data found for requested subjects... Returning \n');
    return
end
params = csv.inputValidation(varargin{:});

blkDates = extracted.blkDates;
rigNames = extracted.rigNames;

maxGrid = cellfun(@(x) [length(unique(x.stim_audDiff)) length(unique(x.stim_visDiff))], ...
    extracted.data(extracted.validSubjects>0), 'uni', 0);
maxGrid = max(cell2mat(maxGrid),2);
axesOpt.figureHWRatio = maxGrid(2)/(1.3*maxGrid(1));
axesOpt.btlrMargins = [100 80 60 100];
axesOpt.gapBetweenAxes = [100 40];
axesOpt.totalNumOfAxes = sum(extracted.validSubjects);

plotData = cell(length(extracted.data), 1);
if ~params.noPlot{1}; figure; end
for i = find(extracted.validSubjects)'
    boxPlot.subject = extracted.subject{i};
    boxPlot.xyLabel = {'AuditoryAzimuth'; 'VisualContrast'};
    boxPlot.axisLimits = [0 1];
    boxPlot.colorMap = plts.general.redBlueMap(64);

    if isempty(extracted.data{i}) || extracted.nExp{i} == 1 
        boxPlot.extraInf = [blkDates{i}{1} ' on ' rigNames{i}{1}];
    else
        boxPlot.extraInf = [num2str(extracted.nExp{i}) ' Sess'];
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
        percentTimeouts = round(mean(tDat.response_direction==0)*100);
        tDat = filterStructRows(tDat, keepIdx);

        [~,~,vLabel] = unique(tDat.stim_visDiff);
        [aUni,~,aLabel] = unique(tDat.stim_audDiff);

        if isfield(tDat, 'block_currentBlock')
            aLabel = aLabel + numel(aUni) * floor((tDat.block_currentBlock-0.5)/2);
            yValues = repmat(aUni, [numel(unique(floor((tDat.block_currentBlock-0.5)/2))), 1]);
        else
            yValues = aUni;
        end

        boxPlot.plotData = accumarray([aLabel, vLabel],tDat.response_direction,[],@mean)-1;
        boxPlot.trialCount = accumarray([aLabel, vLabel],~isnan(tDat.response_direction),[],@sum);
        boxPlot.plotData(boxPlot.trialCount==0) = nan;
        boxPlot.totTrials = length(tDat.stim_visDiff);
        boxPlot.xyValues = {unique(tDat.stim_visDiff)*100; yValues};
        colorBar.colorLabel = 'Fraction of right turns';
        colorBar.colorDirection = 'normal';
        colorBar.colorYTick = {'0'; '1'};
    end
    if ~params.noPlot{1}
        boxPlot.extraInf = [boxPlot.extraInf ', T0:' num2str(percentTimeouts) '%'];
        plts.general.getAxes(axesOpt, find(find(extracted.validSubjects)'==i));
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
    txtD = num2cell([vectorize(xPnts(tIdx)), vectorize(yPnts(tIdx)), vectorize(round(100*plotData(tIdx))/100), vectorize(triNum(tIdx))],2);
    cellfun(@(x) text(x(1),x(2), {num2str(x(3)), num2str(x(4))}, 'horizontalalignment', 'center'), txtD)
end

set(gca, 'xTick', 1:size(plotData,2), 'xTickLabel', boxPlot.xyValues{1}, 'fontsize', 14)
set(gca, 'yTick', 1:size(plotData,1), 'yTickLabel', boxPlot.xyValues{2}, 'fontsize', 14, 'TickLength', [0, 0])
title(sprintf('%s: %d Tri, %s', boxPlot.subject, boxPlot.totTrials, boxPlot.extraInf), 'fontsize', 11)
box off;
end

function v = vectorize(x)
v = x(:);
end
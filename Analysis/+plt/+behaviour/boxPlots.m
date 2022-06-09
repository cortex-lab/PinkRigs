function plotData = boxPlots(varargin)
%% Generate box plots for the behaviour of a mouse/mice
%% Input validation and default assingment
varargin = ['sepPlots', {1}, varargin];
varargin = ['expDef', {'t'}, varargin];
varargin = ['plotType', {'res'}, varargin];
varargin = ['noPlot', {0}, varargin];
params = csv.inputValidation(varargin{:});

nSubjects = length(params.subject);
if nSubjects > 1 && params.sepPlots{1}==1
    fprintf('Multiple subjects, so will combine within subjects \n');
    params.sepPlots = repmat({0},nSubjects,1);
end

[blkDates, rigNames] = deal(cell(nSubjects,1));
expList = csv.queryExp(params);
extractedData = cell(nSubjects,1);
for i = 1:nSubjects
    currData = expList(strcmp(expList.subject, params.subject{i}),:);
    alignedBlock = cellfun(@(x) strcmp(x(1), '1'), currData.alignBlkFrontSideEyeMicEphys);
    if any(~alignedBlock)
        fprintf('Missing block alignments. Will try and align...\n')
        preproc.align.main(currData(~alignedBlock,:), 'process', 'block');
    end

    evExtracted = cellfun(@(x) strcmp(x(end), '1'), currData.preProcSpkEV);
    if any(~evExtracted)
        fprintf('EV extractions. Will try to extract...\n')
        preproc.extractExpData(currData(~evExtracted,:), 'process', 'ev');
    end
    
    currData = csv.queryExp(currData);
    alignedBlock = cellfun(@(x) strcmp(x(1), '1'), currData.alignBlkFrontSideEyeMicEphys);
    evExtracted = cellfun(@(x) strcmp(x(end), '1'), currData.preProcSpkEV);

    failIdx = any(~[alignedBlock, evExtracted],2);
    if any(failIdx)
        failNames = currData.expFolder(failIdx);
        cellfun(@(x) fprintf('WARNING: Files mising for %s. Skipping...\n', x), failNames);
        currData = currData(~failIdx,:);
    end
    if isempty(currData); continue; end

    if length(unique(currData.expDate)) ~= length(currData.expDate)
        expDurations = cellfun(@str2double, currData.expDuration);
        [~, ~, uniIdx] = unique(currData.expDate);
        keepIdx = arrayfun(@(x) find(expDurations == max(expDurations(x == uniIdx))), unique(uniIdx));
        currData = currData(keepIdx,:);
    end
    blkDates{i} = currData.expDate;
    rigNames{i} = strrep(currData.rigName, 'zelda-stim', 'Z');

    loadedEV = csv.loadData(currData, 'loadTag', 'ev');
    evData = [loadedEV.evData{:}];

    for j = 1:length(evData)
        evData(j).stim_visAzimuth(isnan(evData(j).stim_visAzimuth)) = 0;
        evData(j).stim_visDiff = evData(j).stim_visContrast.*sign(evData(j).stim_visAzimuth);
        evData(j).stim_audDiff = evData(j).stim_audAzimuth;
        evData(j).AVParams = unique([evData(j).stim_audDiff evData(j).stim_visDiff], 'rows');
    end

    if ~params.sepPlots{i}
        [uniParams, ~, uniMode] = unique(arrayfun(@(x) num2str(x.AVParams(:)'), evData, 'uni', 0));
        modeIdx = uniMode == mode(uniMode);
        if numel(uniParams) ~= 1
            fprintf('Multiple param sets detected for %s, using mode \n', currData.subject{1});
        end
        names = fieldnames(evData);
        cellData = cellfun(@(f) {vertcat(evData(modeIdx).(f))}, names);

        extractedData{i,1} = cell2struct(cellData, names);
        extractedData{i,1}.nExperiments = sum(modeIdx);
        blkDates{i} = blkDates{i}(modeIdx);
        rigNames{i} = rigNames{i}(modeIdx);
    else
        extractedData = arrayfun(@(x) x,evData, 'uni', 0)';
    end
end
%%
maxGrid = arrayfun(@(x) [length(unique(x.stim_audDiff)) length(unique(x.stim_visDiff))], evData, 'uni', 0);
maxGrid = max(cell2mat(maxGrid),2);
axesOpt.figureHWRatio = maxGrid(2)/(1.3*maxGrid(1));
axesOpt.btlrMargins = [100 80 60 100];
axesOpt.gapBetweenAxes = [100 40];
axesOpt.totalNumOfAxes = length(extractedData);

plotData = cell(length(extractedData), 1);
if ~params.noPlot{1}; figure; end
for i = 1:length(extractedData)
    if isfield(extractedData{i}, 'nExperiments')
        if extractedData{i}.nExperiments == 1
            boxPlot.extraInf = [blkDates{i}{1} ' on ' rigNames{i}{1}];
        end
        tDat = rmfield(extractedData{i}, {'nExperiments', 'AVParams'});
        boxPlot.nExperiments = extractedData{i}.nExperiments;
        boxPlot.subject = params.subject{i};
    else
        tDat = rmfield(extractedData{i}, 'AVParams');
        boxPlot.nExperiments = 1;
        boxPlot.extraInf = [blkDates{1}{i} ' on ' rigNames{1}{i}];
        boxPlot.subject = params.subject{1};
    end


    boxPlot.xyValues = {unique(tDat.stim_visDiff)*100; unique(tDat.stim_audDiff)};
    boxPlot.xyLabel = {'AuditoryAzimuth'; 'VisualContrast'};
    boxPlot.axisLimits = [0 1];
    boxPlot.colorMap = plt.general.redBlueMap(64);

    switch lower(params.plotType{1}(1:3))
        case 'res'
            tkIdx = extractedData{i}.is_validTrial & extractedData{i}.response_direction;
            tDat = filterStructRows(tDat, tkIdx);

            [~,~,vLabel] = unique(tDat.stim_visDiff);
            [~,~,aLabel] = unique(tDat.stim_audDiff);
            boxPlot.plotData = accumarray([aLabel, vLabel],tDat.response_direction,[],@mean)-1;
            boxPlot.trialCount = accumarray([aLabel, vLabel],~isnan(tDat.response_direction),[],@sum);
            boxPlot.plotData(boxPlot.trialCount==0) = nan;
            boxPlot.totTrials = length(tDat.stim_visDiff);
            colorBar.colorLabel = 'Fraction of right turns';
            colorBar.colorDirection = 'normal';
            colorBar.colorYTick = {'0'; '1'};
    end
    if ~params.noPlot{1}
        plt.general.getAxes(axesOpt, i);
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
if isfield(boxPlot, 'extraInf')
    title(sprintf('%s: %d Tri, %s', boxPlot.subject, boxPlot.totTrials, boxPlot.extraInf))
else, title(sprintf('%s: %d Tri, %d Sess', boxPlot.subject, boxPlot.totTrials, boxPlot.nExperiments));
end
set(gca, 'xTick', 1:size(plotData,2), 'xTickLabel', boxPlot.xyValues{1}, 'fontsize', 14)
set(gca, 'yTick', 1:size(plotData,1), 'yTickLabel', boxPlot.xyValues{2}, 'fontsize', 14, 'TickLength', [0, 0])
box off;
end
function plotData = boxPlots(varargin)
%% Generate box plots for the behaviour of a mouse/mice
%% Input validation and default assingment
opt = csv.inputValidation(varargin{:});
opt = csv.addDefaultParam(opt, 'noPlot', 0);
opt = csv.addDefaultParam(opt, 'plotType', 'res');

nSubjects = length(opt.subject);
if nSubjects > 1 && ~isfield(opt, 'sepPlots')
    fprintf('Multiple subjects, so will combine within subjects \n');
    opt = csv.addDefaultParam(opt, 'sepPlots', 0);
elseif ~isfield(opt, 'sepPlots')
    opt = csv.addDefaultParam(opt, 'sepPlots', 1);
end

expList = csv.loadData(opt, loadTag='blk');
expList = expList(cellfun(@isstruct, expList.blkData),:);
%%
[blkDates, rigNames] = deal(cell(nSubjects,1));
for i = 1:nSubjects
    currData = expList(strcmp(expList.subject, opt.subject{i}),:);
    if length(unique(currData.expDate)) ~= length(currData.expDate)
        expDurations = cellfun(@str2double, currData.expDuration);
        [~, ~, uniIdx] = unique(currData.expDate);
        keepIdx = arrayfun(@(x) find(expDurations == max(expDurations(x == uniIdx))), unique(uniIdx));
        currData = currData(keepIdx,:);
    end
    blks = currData.blkData;
    blkDates{i} = currData.expDate;
    rigNames{i} = strrep(currData.rigName, 'zelda-stim', 'Z');

    paramSets = cellfun(@(x) x.events.selected_paramsetValues, blks, 'uni', 0);
    AVParams = cellfun(@(x) [x.audInitialAzimuth(x.numRepeats>0) x.visContrast(x.numRepeats>0)], paramSets, 'uni', 0);
    AVParams = cellfun(@(x) unique([x; x*-1], 'rows'), AVParams, 'uni', 0);

    eventNames = {'visInitialAzimuth'; 'audInitialAzimuth'; 'repeatNum'; ...
        'visContrast'; 'feedback'; 'correctResponse'};
    taskEvents = extractEventsFromBlock(blks, eventNames);

    if ~opt.sepPlots{i}
        [uniParams, ~, uniMode] = unique(cellfun(@(x) num2str(x(:)'), AVParams, 'uni', 0));
        modeIdx = uniMode == mode(uniMode);
        if numel(uniParams) ~= 1
            fprintf('Multiple param sets detected for %s, using mode \n', currData.subject{1});
        end
        names = fieldnames(taskEvents);
        cellData = cellfun(@(f) {vertcat(taskEvents(modeIdx).(f))}, names);

        taskEvents = cell2struct(cellData, names);
        taskEvents.nExperiments = sum(modeIdx);
        blkDates{i} = blkDates{i}(modeIdx);
        rigNames{i} = rigNames{i}(modeIdx);
    end

    for j = 1:length(taskEvents)
        responseRecorded = double(taskEvents(j).correctResponse).*~(taskEvents(j).feedback==0);
        responseRecorded(taskEvents(j).feedback<0) = -1*(responseRecorded(taskEvents(j).feedback<0));
        responseRecorded = ((responseRecorded>0)+1).*(responseRecorded~=0);
        taskEvents(j).responseRecorded = responseRecorded;

        taskEvents(j).visDiff = taskEvents(j).visContrast.*sign(taskEvents(j).visInitialAzimuth);
        taskEvents(j).audDiff = taskEvents(j).audInitialAzimuth;
    end

    if i == 1; extractedData = taskEvents;
    else, extractedData(i,1) = taskEvents;
    end
    
    % remove empty experiments
    emptyExpIdx = cellfun(@(x) isempty(x), {extractedData.visInitialAzimuth});
    extractedData(emptyExpIdx) = [];
    blkDates(emptyExpIdx) = [];
    rigNames(emptyExpIdx) = [];
end
%%
maxGrid = arrayfun(@(x) [length(unique(x.audDiff)) length(unique(x.visDiff))], extractedData, 'uni', 0);
maxGrid = max(cell2mat(maxGrid),2);
axesOpt.figureHWRatio = maxGrid(2)/(1.3*maxGrid(1));
axesOpt.btlrMargins = [100 80 60 100];
axesOpt.gapBetweenAxes = [100 40];
axesOpt.totalNumOfAxes = length(extractedData);

plotData = cell(length(extractedData), 1);
if ~opt.noPlot{1}; figure; end
for i = 1:length(extractedData)
    if isfield(extractedData(i), 'nExperiments')
        if extractedData(i).nExperiments == 1
            boxPlot.extraInf = [blkDates{i}{1} ' on ' rigNames{i}{1}];
        end
        tDat = rmfield(extractedData(i), 'nExperiments');
        boxPlot.nExperiments = extractedData(i).nExperiments;
    else
        tDat = extractedData(i);
        boxPlot.nExperiments = 1;
        boxPlot.extraInf = [blkDates{i}{1} ' on ' rigNames{i}{1}];
    end

    boxPlot.subject = currData.subject{1};

    boxPlot.xyValues = {unique(tDat.visDiff)*100; unique(tDat.audDiff)};
    boxPlot.xyLabel = {'AuditoryAzimuth'; 'VisualContrast'};
    boxPlot.axisLimits = [0 1];
    boxPlot.colorMap = plt.general.redBlueMap(64);

    switch lower(opt.plotType{1}(1:3))
        case 'res'
            timeOuts = extractedData(i).responseRecorded == 0;
            repTrials = timeOuts*0;
            repNums = extractedData(i).repeatNum;
            rep1 = find(repNums==1);
            for j = 1:length(extractedData(i).responseRecorded)
                if repNums(j)~=1 && ~timeOuts(j)
                    tstIdx = rep1(find(rep1<j,1,'last'));
                    if sum(~timeOuts(tstIdx:j)) ~= 1
                        repTrials(j) = 1;
                    end
                end
            end

            tDat = filterStructRows(tDat, ~timeOuts & ~repTrials);

            [~,~,vLabel] = unique(tDat.visDiff);
            [~,~,aLabel] = unique(tDat.audDiff);
            boxPlot.plotData = accumarray([aLabel, vLabel],tDat.responseRecorded,[],@mean)-1;
            boxPlot.trialCount = accumarray([aLabel, vLabel],~isnan(tDat.responseRecorded),[],@sum);
            boxPlot.plotData(boxPlot.trialCount==0) = nan;
            boxPlot.totTrials = length(tDat.visDiff);
            colorBar.colorLabel = 'Fraction of right turns';
            colorBar.colorDirection = 'normal';
            colorBar.colorYTick = {'0'; '1'};
    end
    if ~opt.noPlot{1}
        plt.general.getAxes(axesOpt, i);
        makePlot(boxPlot);
    end
    plotData{i,1} = boxPlot;
end
if ~opt.noPlot{1}
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
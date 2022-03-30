function boxPlots(subject, expDate, plotType, sepPlots, expDef)
%% Generate box plots for the behaviour of a mouse/mice

% INPUTS (default)
% subject---Subject(s) to plt. Can be cell array (must provide)

% expDate---expDate(s) to plot (last5)
    %        'yyyy-mm-dd'--------------A specific date
    %        'all'---------------------All data
    %        'lastx'-------------------The last x days of data (especially useful during training)
    %        'firstx'------------------The first x days of data
    %        'yest'--------------------The x-1 day, where x is the most recent day
    %        'yyyy-mm-dd:yyyy-mm-dd'---Load dates in this range (including the boundaries)
    
% plotType--Type of box plot ('res')
    %        'res'---------------------Fraction of rightward responses
    %        'tim'---------------------Median reaction times for each condition
    
% sepPlots--Separate plots by session--only possible if 1 subject (1)

% expDef----Name of required expDef ('multiSpaceWorld_checker_training')

%% Input validation and default assingment
figure;
if ~exist('subject', 'var'); error('Must specify subject'); end
if ~exist('expDate', 'var'); expDate = 'last5'; end
if ~exist('plotType', 'var'); plotType = 'res'; end
if ~exist('sepPlots', 'var'); sepPlots = 1; end
if ~exist('expDef', 'var'); expDef = 'multiSpaceWorld_checker_training'; end
if ~iscell(subject); subject = {subject}; end

if length(subject) > 1
    fprintf('Multiple subjects, so will combine within subjects \n');
    sepPlots = 0;
end

blkDates = cell(length(subject),1);
rigNames = cell(length(subject),1);

%%
for i = 1:length(subject)
    [blks, extractedDates] = getDataFromDates(subject{i}, expDate, 'any', expDef);
    if length(unique(extractedDates)) ~= length(extractedDates)
        expDurations = cellfun(@(x) x.duration, blks);
        [~, ~, uniIdx] = unique(extractedDates);
        keepIdx = arrayfun(@(x) find(expDurations == max(expDurations(x == uniIdx))), unique(uniIdx));
        blks = blks(keepIdx);
    end
    blkDates{i} = cellfun(@(x) datestr(x.endDateTime, 'yyyy-mm-dd'), blks, 'uni', 0);
    rigNames{i} = cellfun(@(x) [upper(x.rigName(1)) x.rigName(end)], blks, 'uni', 0);
    
    paramSets = cellfun(@(x) x.events.selected_paramsetValues, blks, 'uni', 0);
    AVParams = cellfun(@(x) [x.audInitialAzimuth(x.numRepeats>0) x.visContrast(x.numRepeats>0)], paramSets, 'uni', 0);
    AVParams = cellfun(@(x) unique([x; x*-1], 'rows'), AVParams, 'uni', 0);
    
    eventNames = {'visInitialAzimuth'; 'audInitialAzimuth'; 'repeatNum'; ...
        'visContrast'; 'feedback'; 'correctResponse'};
    taskEvents = extractEventsFromBlock(blks, eventNames);
    
    if ~sepPlots
        [uniParams, ~, uniMode] = unique(cellfun(@(x) num2str(x(:)'), AVParams, 'uni', 0));
        modeIdx = uniMode == mode(uniMode);
        if numel(uniParams) ~= 1
            fprintf('Multiple param sets detected for %s, using mode \n', subject{i});
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
end
%%
maxGrid = arrayfun(@(x) [length(unique(x.audDiff)) length(unique(x.visDiff))], extractedData, 'uni', 0);
maxGrid = max(cell2mat(maxGrid),2);
axesOpt.figureHWRatio = maxGrid(2)/(1.3*maxGrid(1));
axesOpt.btlrMargins = [100 80 60 100];
axesOpt.gapBetweenAxes = [100 40];
axesOpt.totalNumOfAxes = length(extractedData);

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
        boxPlot.extraInf = [blkDates{1}{i} ' on ' rigNames{1}{i}];
    end
    
    if length(subject) == 1
        boxPlot.subject = subject;
    else, boxPlot.subject = subject(i);
    end
    
    boxPlot.xyValues = {unique(tDat.visDiff)*100; unique(tDat.audDiff)};
    boxPlot.xyLabel = {'AuditoryAzimuth'; 'VisualContrast'};
    boxPlot.axisLimits = [0 1];
    boxPlot.colorMap = plt.general.redBlueMap(64);
    
    switch lower(plotType(1:3))
        case 'res'
            timeOuts = extractedData(i).responseRecorded == 0;
            tDat = filterStructRows(tDat, ~timeOuts);
            [~,~,vLabel] = unique(tDat.visDiff);
            [~,~,aLabel] = unique(tDat.audDiff);
            boxPlot.plotData = accumarray([aLabel, vLabel],tDat.responseRecorded,[],@mean)-1;
            boxPlot.trialCount = accumarray([aLabel, vLabel],tDat.responseRecorded,[],@sum);
            boxPlot.plotData(boxPlot.trialCount==0) = nan;
            boxPlot.totTrials = length(tDat.visDiff);
            colorBar.colorLabel = 'Fraction of right turns';
            colorBar.colorDirection = 'normal';
            colorBar.colorYTick = {'0'; '1'};
    end
    plt.general.getAxes(axesOpt, i);
    makePlot(boxPlot);
end
currentAxisPotision = get(gca, 'position');
figureSize = get(gcf, 'position');

colorBar.handle = colorbar;
set(colorBar.handle, 'Ticks', get(colorBar.handle, 'Limits'), 'TickLabels', colorBar.colorYTick, 'YDir', colorBar.colorDirection);
set(gca, 'position', currentAxisPotision);
colorBar.textHandle = ylabel(colorBar.handle, colorBar.colorLabel);
set(colorBar.textHandle, 'position', [1 mean(get(colorBar.handle, 'Limits')) 0], 'FontSize', 14)
set(colorBar.handle, 'position', [1-75/figureSize(3), 0.2, 30/figureSize(3), 0.6])
end


function makePlot(boxPlot, addText)
if ~exist('addText', 'var'); addText = 1; end
if ~isfield(boxPlot, 'plotLabels'); boxPlot.plotLabels = boxPlot.plotData; end
if iscell(boxPlot.subject); boxPlot.subject = boxPlot.subject{1}; end
plotData = boxPlot.plotData;
imAlpha=ones(size(plotData));
imAlpha(isnan(plotData))=0;
imagesc(plotData, 'AlphaData', imAlpha);
caxis(boxPlot.axisLimits);
colormap(boxPlot.colorMap);
daspect([1 1 1]); axis xy;
[xPnts, yPnts] = meshgrid(1:size(plotData,2), 1:size(plotData,1));
if addText
    txtD = num2cell([xPnts(~isnan(plotData)), yPnts(~isnan(plotData)), plotData(~isnan(plotData))],2);
    cellfun(@(x) text(x(1),x(2), num2str(round(x(3)*100)/100), 'horizontalalignment', 'center'), txtD)
end
if isfield(boxPlot, 'extraInf')
    title(sprintf('%s: %d Tri, %s', boxPlot.subject, boxPlot.totTrials, boxPlot.extraInf))
else, title(sprintf('%s: %d Tri, %d Sess', boxPlot.subject, boxPlot.totTrials, boxPlot.nExperiments));
end
set(gca, 'xTick', 1:size(plotData,2), 'xTickLabel', boxPlot.xyValues{1}, 'fontsize', 14)
set(gca, 'yTick', 1:size(plotData,1), 'yTickLabel', boxPlot.xyValues{2}, 'fontsize', 14, 'TickLength', [0, 0])
box off;
end
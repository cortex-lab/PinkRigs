function plotData = glmFit(subject, expDate, plotType, expDef, opt)
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
nSubjects = length(subject);
if exist('opt', 'var')
    if isfield(opt, 'sepPlots'); sepPlots = opt.sepPlots; end
    if isfield(opt, 'expNum'); expNum = opt.expNum; end
    if isfield(opt, 'noPlot'); noPlot = opt.noPlot; end
end

if ~exist('subject', 'var'); error('Must specify subject'); end
if ~exist('expDate', 'var'); expDate = 'last5'; end
if ~exist('expNum', 'var'); expNum = 'any'; end
if ~exist('plotType', 'var'); plotType = 'res'; end
if ~exist('expDef', 'var'); expDef = 'multiSpaceWorld_checker_training'; end
if ~iscell(subject); subject = {subject}; end
if ~exist('noPlot', 'var'); noPlot = 0; end

if ~strcmp(expNum, 'any')
    if ~all([length(expNum) length(expDate)] == nSubjects)
        error('If requesting expNum, subjects/date/expnum must be the same size');
    end
end

if nSubjects > 1 && ~exist('sepPlots', 'var')
    fprintf('Multiple subjects, so will combine within subjects \n');
    sepPlots = 0;
elseif ~exist('sepPlots', 'var')
    sepPlots = 1;
end

if ~iscell(expDate); expDate = {expDate}; end
if ~iscell(expNum); expNum = {expNum}; end
if ~iscell(expDef); expDef = {expDef}; end


if length(expDate)==1 && 1 ~= nSubjects; expDate = repmat(expDate, nSubjects); end
if length(expNum)==1 && 1 ~= nSubjects; expNum = repmat(expNum, nSubjects); end
if length(expDef)==1 && 1 ~= nSubjects; expDef = repmat(expDef, nSubjects); end

blkDates = cell(nSubjects,1);
rigNames = cell(nSubjects,1);

%%
for i = 1:nSubjects
    [blks, extractedDates] = getDataFromDates(subject{i}, expDate{i}, expNum{i}, expDef{1});
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

plotData = cell(length(extractedData), 1);
if ~noPlot; figure; end
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
    
    if nSubjects == 1
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
    if ~noPlot
        plt.general.getAxes(axesOpt, i);
        makePlot(boxPlot);
    end
    plotData{i,1} = boxPlot;
end
if ~noPlot
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
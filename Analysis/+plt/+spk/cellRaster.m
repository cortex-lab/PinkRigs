function cellRaster(dat)
%% Cell raster "browser"
% NOTE: Designed to operate on the output of the PinkAV Rigs pipeline.
% Letters below are used as follows:
%    "p": number of probes
%    "s": number of spikes
%    "c": number of clusters
%    "n": number of sets of evTimes
%    "m": number of evTimes within a set
%    "q": number of distinct trial groupings for each set of event times


% INPUTS(default values)
% spk(Required)-----------[px1] cell of structs: spike info for p probes in the same recording
%   spk{1}.spikes.time-------------[sx1] matrix: time of each spike
%   spk{1}.spikes.cluster----------[sx1] matrix: cluster ID for each spike
%   spk{1}.spikes.tempScalingAmp---[sx1] matrix: scaled amplitude for each spike
%   spk{1}.clusters.ID-------------[cx1] matrix: ID for each cluster
%   spk{1}.clusters.Depth----------[cx1] matrix: Distance form probe tip for each cluster
%   spk{1}.clusters.XPos-----------[cx1] matrix: x position for each cluster (0 = shank 0)

% evTimes(Required)-------[nx1] cell array of [mx1] matrices: "n" sets event times
%   ALTERNATIVELY can be "ev" from the preproc file. See "plotOpt.customPipe" below
%   NOTE: "nans" will be automatically removed

% triGrps(ones(m,1))-----[nx1] cell array of [mxq] matrices: "q" sets of trial labels for events
%   NOTE: A uniform label (ones(p,1)) will be added to the first column if absent

% plotOpt-------Struct with optional inputs. Each of these will have default values
%
%    .paramTag('default')-----string: identifier for param set in combination with "ev" input
%       NOTE: custom paramSets can be stored in +rasterParams folder. paramTag is function name
%
%    .groupNames('Unsorted')------[nx1] cell array of [1xq] cell arrays: names for each trial group
%       NOTE: each cell array should contain a name for each unique value within that trial group
%
%    .sortTemplates('sig')------string OR [nx1] cell array of [cx1] matrices: how to sort clusters
%       NOTE: This only affects the order of cycling through clusters (with up/down arrow)
%
%    .sortTrials(evTimes)------[nx1] cell array of [mx1] matrices: indicates how to sort trials
%       NOTE: Will sort trials within each trial group according to increasing values in .sortTrials
%
%    .trialTickTimes([])--[nx1] cell array of [mx1] matrices: additional raster ticks (e.g. movements)
%
%    .highlight([])-------[nx1] cell array of [cx1] logical matrices: visually highligts indicated clusters


%% Controls:
% Up/Down arrow keys:             switch between different clusters
% Left/Right arrow keys:          switch between different groups of trial labels
% Shift+Left/Right arrow keys:    switch between different sets of event times
% "c" key:                        manually select a cluster

%% Package gui data
cellrasterGui = figure('color','w');
guiData = struct;
guiData.titleMain = annotation('textbox', [0, 1, 1, 0], 'string', 'My Text', 'EdgeColor', 'none',...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
defFontSize = 8;


%% Set up the cluster dots axes
nCol = 15;
nRow = 15;

axWidth = 1:2;
axHeight = 1:nRow;
tRef = (repmat(nRow*(axHeight-1)', 1, length(axWidth)) + repmat(axWidth,length(axHeight), 1))';
guiData.axes.clusters = subplot(nRow,nCol,tRef(:)','YDir','normal','ButtonDownFcn',@clusterClick); hold on; axis tight
xlim([-0.1,1]);
ylabel('Distance from tip (\mum)', 'FontSize',defFontSize)
xlabel('xPosition (\mum)', 'FontSize',defFontSize)
disableDefaultInteractivity(gca)

%% Set up the psth axes
axWidth = 4:9;
axHeight = 1:3;
tRef = (repmat(nRow*(axHeight-1)', 1, length(axWidth)) + repmat(axWidth,length(axHeight), 1))';
guiData.axes.psth(1) = subplot(nRow,nCol,tRef(:)','YAxisLocation','left'); hold on; axis tight
set(guiData.axes.psth(1), 'XColor', 'none', 'position', get(guiData.axes.psth(1), 'position').*[1 0.95 1 0.95])
ylabel('Spks/s', 'FontSize',defFontSize);
disableDefaultInteractivity(gca)

axWidth = 10:nCol;
axHeight = 1:3;
tRef = (repmat(nRow*(axHeight-1)', 1, length(axWidth)) + repmat(axWidth,length(axHeight), 1))';
guiData.axes.psth(2) = subplot(nRow,nCol,tRef(:)','YAxisLocation','right'); hold on; axis tight
set(guiData.axes.psth(2), 'XColor', 'none', 'position', get(guiData.axes.psth(2), 'position').*[1 0.95 1 0.95])
ylabel('Spks/s', 'FontSize',defFontSize);
disableDefaultInteractivity(gca)

guiData.axes.yLimListener(1) = addlistener(guiData.axes.psth(1), 'YLim', 'PostSet', ...
    @(cellrasterGui,eventdata) matchYLim(guiData.axes.psth(1), guiData.axes.psth(2)));
guiData.axes.yLimListener(2) = addlistener(guiData.axes.psth(2), 'YLim', 'PostSet', ...
    @(cellrasterGui,eventdata) matchYLim(guiData.axes.psth(2), guiData.axes.psth(1)));
%% Set up the raster axes
axWidth = 4:9;
axHeight = round(nRow*0.3):round(nRow*0.7);
tRef = (repmat(nRow*(axHeight-1)', 1, length(axWidth)) + repmat(axWidth,length(axHeight), 1))';
guiData.axes.raster(1) = subplot(nRow,nCol,tRef(:)', ...
    'YDir','reverse','YAxisLocation','left');
hold on; axis tight
ylabel('Trial', 'FontSize',defFontSize);
disableDefaultInteractivity(gca)

axWidth = 10:nCol;
tRef = (repmat(nRow*(axHeight-1)', 1, length(axWidth)) + repmat(axWidth,length(axHeight), 1))';
guiData.axes.raster(2) = subplot(nRow,nCol,tRef(:)', ...
    'YDir','reverse','YAxisLocation','right');
hold on; axis tight
ylabel('Trial', 'FontSize',defFontSize);
disableDefaultInteractivity(gca)
linkaxes([guiData.axes.psth(:);guiData.axes.raster(:)], 'x');

%% Set up the amplitude axes
axHeight = nRow-1:nRow;
axWidth = 4:9;
tRef = (repmat(nRow*(axHeight-1)', 1, length(axWidth)) + repmat(axWidth,length(axHeight), 1))';
guiData.axes.amplitude(1) = subplot(nRow,nCol,tRef(:)');
hold on; axis tight
ylabel('Amplitude', 'FontSize',defFontSize);
disableDefaultInteractivity(gca)

axWidth = 10:nCol;
tRef = (repmat(nRow*(axHeight-1)', 1, length(axWidth)) + repmat(axWidth,length(axHeight), 1))';
guiData.axes.amplitude(2) = subplot(nRow,nCol,tRef(:)','YAxisLocation','right');
hold on; axis tight
ylabel('Amplitude', 'FontSize',defFontSize);
disableDefaultInteractivity(gca)
%%

dualPosition = cell2mat(get(guiData.axes.raster(:), 'position'));
guiData.axes.fullWidth = [dualPosition(1,1) dualPosition(2,1)-dualPosition(1,1)+dualPosition(2,3)];
guiData.axes.halfWidth = dualPosition(2,[1,3]);
guiData.axes.modPos = @(x,y) set(x, 'position', get(x, 'position').*[0 1 0 1]+[y(1) 0 y(2) 0]);

guiData.axes.rasterLabel = annotation('textbox', 'EdgeColor', 'none', 'BackgroundColor', 'none',...
    'string', 'Time from event (s)', 'FontSize', defFontSize, 'position', mean(dualPosition).*[1 0.93 1 0], ...
    'HorizontalAlignment', 'center');

guiData.axes.amplitdueLabel = annotation('textbox', 'EdgeColor', 'none', 'BackgroundColor', 'none',...
    'string', 'Time (s)', 'FontSize', defFontSize, 'position', mean(dualPosition).*[1 0.25 1 0], ...
    'HorizontalAlignment', 'center');

guiData.titleSub{1} = annotation('textbox', [dualPosition(1,1), 0.95, dualPosition(1,3), 0], 'string', 'My Text', 'EdgeColor', 'none',...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.8500 0.3250 0.0980]);
guiData.titleSub{2} = annotation('textbox', [dualPosition(2,1), 0.95, dualPosition(2,3), 0], 'string', 'My Text', 'EdgeColor', 'none',...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.4940 0.1840 0.5560]);
%% Remove sessions where clusters do not match
% keepIdx = ones(length(dat),1)>0;
% for i = 1:length(dat)
%     spks = dat{i}.spks;
%     if length(spks)==1; continue; end
%     clustCompare = cellfun(@(x) structfun(@(y) length(y.clusters.depths), x), spks, 'uni', 0);
%     if ~all(clustCompare{1}(:)==clustCompare{2}(:))
%         keepIdx(i) = 0;
%         fprintf('WARNING: Data idx %d of %d had sessions sorted separately. Removeing... \n', i, length(dat))
%     end
% end
% length(unique(cellfun(@length, guiData.curr.clusterXPos))) ~= 1

%%
guiData.allData = dat;
guiData.nSessions = length(dat);
guiData.curr.dIdx = 1;
guiData.curr.probe = 1;
guiData.curr.evTimeRef = 1;
guiData.curr.triGrpRef = 1;
guidata(cellrasterGui, guiData);

assignGUIFields(cellrasterGui);
cycleProbe(cellrasterGui);
updatePlot(cellrasterGui);
end

%%
function guiData = assignGUIFields(cellrasterGui, guiData)
if ~exist('guiData', 'var'); guiData = guidata(cellrasterGui); end
dat = guiData.allData;
dIdx = guiData.curr.dIdx;


evTimesLength = cellfun(@(x)length(x), dat{dIdx}.evTimes);
if all(evTimesLength<guiData.curr.evTimeRef); guiData.curr.evTimeRef = 1;
elseif guiData.curr.evTimeRef < 1; guiData.curr.evTimeRef = max(evTimesLength);
end
evTimeRef = guiData.curr.evTimeRef;

triGrpIdx = cellfun(@length, dat{dIdx}.triGrps) >=evTimeRef;
triGrpLength = zeros(length(triGrpIdx),1);
triGrpLength(triGrpIdx) = cellfun(@(x) size(x{evTimeRef},2),  dat{dIdx}.triGrps(triGrpIdx));
if all(triGrpLength<guiData.curr.triGrpRef); guiData.curr.triGrpRef = 1;
elseif guiData.curr.triGrpRef < 1; guiData.curr.triGrpRef = max(triGrpLength);
end
triGrpRef = guiData.curr.triGrpRef;

nExps = all([evTimeRef <= evTimesLength,  guiData.curr.triGrpRef <= triGrpLength],2);
nanIdx = cellfun(@(x) isnan(x{evTimeRef}), dat{dIdx}.evTimes(nExps), 'uni', 0);

guiData.curr.nExps = sum(nExps);
guiData.curr.subject = unique(dat{dIdx}.subject(nExps), 'stable');
guiData.curr.expDate = unique(dat{dIdx}.expDate(nExps), 'stable');
guiData.curr.expDef = unique(dat{dIdx}.plotName(nExps), 'stable');
guiData.curr.evTimes = cellfun(@(x,y) x{evTimeRef}(~y), dat{dIdx}.evTimes(nExps), nanIdx, 'uni', 0);
guiData.curr.evNames = cellfun(@(x) x{evTimeRef}, dat{dIdx}.evNames(nExps), 'uni', 0);

guiData.curr.triGrps = cellfun(@(x,y) x{evTimeRef}(~y, triGrpRef), dat{dIdx}.triGrps(nExps), nanIdx, 'uni', 0);
guiData.curr.grpNames = cellfun(@(x) x{evTimeRef}{triGrpRef}, dat{dIdx}.grpNames(nExps), 'uni', 0);

probeFields = fields([dat{dIdx}.spks{nExps}]);
selProbe = probeFields{guiData.curr.probe};
guiData.curr.spks = [dat{dIdx}.spks{nExps}]';
guiData.curr.nProbes = length(probeFields);
guiData.curr.spks = {guiData.curr.spks.(selProbe)}';

guiData.curr.spkTimes = cellfun(@(x) x.spikes.times, guiData.curr.spks, 'uni', 0);
guiData.curr.spkCluster = cellfun(@(x) x.spikes.clusters, guiData.curr.spks, 'uni', 0);
guiData.curr.spkAmps = cellfun(@(x) x.spikes.amps, guiData.curr.spks, 'uni', 0);
guiData.curr.clusterDepths = cellfun(@(x) x.clusters.depths', guiData.curr.spks, 'uni', 0);
guiData.curr.clusterXPos = cellfun(@(x) x.clusters.xpos', guiData.curr.spks, 'uni', 0);
guiData.curr.clusterIDs = cellfun(@(x) x.clusters.IDs', guiData.curr.spks, 'uni', 0);

guiData.curr.sortTemplates = dat{dIdx}.sortTemplates(find(nExps,1));

sortExists = cellfun(@(x) ~isempty(x), dat{dIdx}.sortTrials);
guiData.curr.sortTrials = cell(sum(nExps),1);
guiData.curr.trialTickTimes = cell(sum(nExps),1);
guiData.curr.sortTrials(sortExists) = cellfun(@(x,y) x{evTimeRef}(~y), dat{dIdx}.sortTrials(nExps&sortExists), nanIdx(sortExists), 'uni', 0);
guiData.curr.trialTickTimes(sortExists) = cellfun(@(x,y) x{evTimeRef}(~y), dat{dIdx}.trialTickTimes(nExps&sortExists), nanIdx(sortExists), 'uni', 0);
guiData.curr.highlight = cell(sum(nExps), 1);


if length(unique(cellfun(@length, guiData.curr.clusterXPos))) ~= 1
    error('Multiple sessions must be from single sorting, but temaplates are different ?! \n')
else
    guiData.curr.clusterXPos = guiData.curr.clusterXPos{1};
    guiData.curr.clusterDepths = guiData.curr.clusterDepths{1};
    guiData.curr.clusterIDs = guiData.curr.clusterIDs{1};
end

axHand1 = [guiData.axes.psth(1); guiData.axes.raster(1); guiData.axes.amplitude(1)];
axHand2 = [guiData.axes.psth(2); guiData.axes.raster(2); guiData.axes.amplitude(2)];
if guiData.curr.nExps == 1
    arrayfun(@(x) set(x, 'visible', 0), axHand1, 'uni', 0);
    arrayfun(@(x) guiData.axes.modPos(x, guiData.axes.fullWidth), axHand2, 'uni', 0);
elseif guiData.curr.nExps == 2
    arrayfun(@(x) set(x, 'visible', 1), axHand1, 'uni', 0);
    arrayfun(@(x) guiData.axes.modPos(x, guiData.axes.halfWidth), axHand2, 'uni', 0);
end

guidata(cellrasterGui, guiData);
end

%%
function cycleProbe(cellrasterGui, changeCluster)
if ~exist('changeCluster', 'var'); changeCluster = 1; end
guiData = guidata(cellrasterGui);

% Find responsive cells
guiData.sigRes = cellfun(@(x,y) neural.findResponsiveCells(x,y), guiData.curr.spks, guiData.curr.evTimes, 'uni', 0);

%Sort clusters (order of cycling with arrow keys) according to optional inputs
% Remove the zero idx (if present) from clusters;
if ischar(guiData.curr.sortTemplates{1})
    switch guiData.curr.sortTemplates{1}(1:3)
        case 'dep'
            [~, clusterSortIdx] = sort(guiData.curr.clusterDepths{1});
        case 'sig'
            [~, chooseSort] = max(cellfun(@(x) sum(x.pVal<0.05), guiData.sigRes));
            [~, clusterSortIdx] = sort(guiData.sigRes{chooseSort}.pVal);
            guiData.curr.highlight = cellfun(@(x) x.pVal<0.0001, guiData.sigRes, 'uni', 0);
    end
else
    [~, clusterSortIdx] = sort(guiData.sortTemplates{guiData.curr.evTimeRef});
end
if changeCluster
    guiData.curr.sortedTemplateIDs = guiData.curr.clusterIDs(clusterSortIdx);
    guiData.curr.cluster = guiData.curr.sortedTemplateIDs(1);
    guiData.curr.clusterIdx = find(guiData.curr.cluster == guiData.curr.clusterIDs,1);
end

% Initialize figure and axes
axes(guiData.axes.clusters); cla;
clusterXPos = guiData.curr.clusterXPos;
clusterDepths = guiData.curr.clusterDepths;

ylim([min(clusterDepths)-50 max(clusterDepths)+50]);
xlim([min(clusterXPos)-5 max(clusterXPos)+5]);

% (plot cluster depths by depth and xPos)
cCol = [[0.8500 0.3250 0.0980];[0.4940 0.1840 0.5560]];
highlight = guiData.curr.highlight;
dualSig = sum(cell2mat(highlight'),2);
for i = 1:guiData.curr.nExps
    faceCol = repmat(cCol(i,:), length(dualSig),1);
    faceCol(dualSig==2,:) = repmat([1,0,1], sum(dualSig==2),1);
    scatter(clusterXPos(highlight{i}), clusterDepths(highlight{i}), 50, faceCol(highlight{i},:),...
        'filled', 'MarkerEdgeColor', 'none');
end
clusterDots = plot(clusterXPos, clusterDepths,'.k','MarkerSize',10,'ButtonDownFcn',@clusterClick);
currClusterDot = plot(0,0,'.r','MarkerSize',30);

[psthLines, rasterDots, rasterOnset, addRasterTicks, amplitudePlot, psthOnset] = deal(cell(2,1));
for i = 1:2
    % (smoothed psth)
    maxNumGroups = 10; %HARDCODED for now... unlikely to be more than 10 trial labels...
    cla(guiData.axes.psth(i));
    psthLines{i} = arrayfun(@(x) plot(guiData.axes.psth(i), NaN,NaN,'linewidth',2,'color','k'),1:maxNumGroups);
    psthOnset{i} = xline(guiData.axes.psth(i), 0, ':c', 'linewidth', 3);

    % (raster)
    cla(guiData.axes.raster(i));
    rasterDots{i} = scatter(guiData.axes.raster(i), NaN,NaN,5,'k','filled');
    rasterOnset{i} = xline(guiData.axes.raster(i), 0, ':c', 'linewidth', 3);
    addRasterTicks{i} = scatter(guiData.axes.raster(i), NaN,NaN,10,'c','filled');

    % (spk amplitude across the recording)
    cla(guiData.axes.amplitude(i));
    amplitudePlot{i} = plot(guiData.axes.amplitude(i), NaN,NaN,'.k');

    % Set default raster times
    rasterWindow = [-0.5,1]./guiData.curr.nExps;
    psthBinSize = 0.001;
    tBins = rasterWindow(1):psthBinSize:rasterWindow(2);
    t = tBins(1:end-1) + diff(tBins)./2;
end

% Set functions for key presses
set(cellrasterGui,'WindowKeyPressFcn',@keyPress);

% (plots)
guiData.plot.clusterDots = clusterDots;
guiData.plot.currClusterDot = currClusterDot;
guiData.plot.psthLines = psthLines;
guiData.plot.psthOnset = psthOnset;
guiData.plot.rasterDots = rasterDots;
guiData.plot.rasterOnset = rasterOnset;
guiData.plot.addRasterTicks = addRasterTicks;
guiData.plot.amplitudes = amplitudePlot;

% (raster times)
guiData.plot.rasterTime = t;
guiData.plot.rasterBins = tBins;

% Upload gui data and draw
guidata(cellrasterGui, guiData);
end

function updatePlot(cellrasterGui)
% Get guidata
guiData = guidata(cellrasterGui);
guiData.curr.clusterIdx = find(guiData.curr.cluster == guiData.curr.clusterIDs);

% Plot depth location on probe
clusterX = get(guiData.plot.clusterDots,'XData');
clusterY = get(guiData.plot.clusterDots,'YData');
set(guiData.plot.currClusterDot,'XData',clusterX(guiData.curr.clusterIdx), 'YData',clusterY(guiData.curr.clusterIdx));

% Bin spks (use only spks within time range, big speed-up)
for i = 1:guiData.curr.nExps
    if guiData.curr.nExps == 2; pltIdx = i; else, pltIdx = 2; end

    currSpkIdx = ismember(typecast(guiData.curr.spkCluster{i},'uint32'),guiData.curr.cluster);
    currRasterSpkTimes = guiData.curr.spkTimes{i}(currSpkIdx);

    tPeriEvent = guiData.curr.evTimes{i} + guiData.plot.rasterBins;
    % (handle NaNs by setting rows with NaN times to 0)
    tPeriEvent(any(isnan(tPeriEvent),2),:) = 0;

    currRasterSpkTimes(currRasterSpkTimes < min(tPeriEvent(:)) | ...
        currRasterSpkTimes > max(tPeriEvent(:))) = [];

    tDat = tPeriEvent';
    if ~any(diff(tDat(:))<0)
        currRaster = [histcounts(currRasterSpkTimes,tDat(:)),0];
        currRaster = reshape(currRaster, size(tPeriEvent'))';
        currRaster(:,end) = [];
    else
        currRaster = cell2mat(arrayfun(@(x) ...
            histcounts(currRasterSpkTimes,tPeriEvent(x,:)), ...
            [1:size(tPeriEvent,1)]','uni',false));
    end

    % Set color scheme
    currGroup = guiData.curr.triGrps{i};
    currAddTicks = guiData.curr.trialTickTimes{i};
    if ~isempty(currAddTicks) && any(currAddTicks)
        if max(currAddTicks) > 100
            currAddTicks = currAddTicks - guiData.curr.evTimes{i};
        end
    end
    if length(unique(currGroup)) == 1
        % Black if one group
        groupColors = [0,0,0];
    elseif length(unique(sign(currGroup(currGroup ~= 0)))) == 1
        % Black-to-red single-signed groups
        nGroups = length(unique(currGroup));
        groupColors = [linspace(0,0.8,nGroups)',zeros(nGroups,1),zeros(nGroups,1)];
    elseif length(unique(sign(currGroup(currGroup ~= 0)))) == 2
        % Symmetrical blue-black-red if negative and positive groups
        nGroupsPos = length(unique(currGroup(currGroup > 0)));
        groupColorsPos = [linspace(0.3,1,nGroupsPos)',zeros(nGroupsPos,1),zeros(nGroupsPos,1)];

        nGroupsNeg = length(unique(currGroup(currGroup < 0)));
        groupColorsNeg = [zeros(nGroupsNeg,1),zeros(nGroupsNeg,1),linspace(0.3,1,nGroupsNeg)'];

        nGroupsZero = length(unique(currGroup(currGroup == 0)));
        groupColorsZero = [zeros(nGroupsZero,1),zeros(nGroupsZero,1),zeros(nGroupsZero,1)];

        groupColors = [flipud(groupColorsNeg);groupColorsZero;groupColorsPos];
    end

    % Plot smoothed PSTH
    gw = gausswin(31,3)';
    smWin = gw./sum(gw);
    currPsth = grpstats(currRaster,currGroup,@(x) mean(x,1));
    padStart = repmat(mean(currPsth(:,1:10),2), 1, floor(length(smWin)/2));
    padEnd = repmat(mean(currPsth(:,end-9:end),2), 1, floor(length(smWin)/2));
    currSmoothedPsth = conv2([padStart,currPsth,padEnd], smWin,'valid')./mean(diff(guiData.plot.rasterTime));

    % (set the first n lines by group, set all others to NaN)
    arrayfun(@(x) set(guiData.plot.psthLines{pltIdx}(x), ...
        'XData',guiData.plot.rasterTime,'YData',currSmoothedPsth(x,:), ...
        'Color',groupColors(x,:)),1:size(currPsth,1));
    arrayfun(@(align_group) set(guiData.plot.psthLines{pltIdx}(align_group), ...
        'XData',NaN,'YData',NaN), ...
        size(currPsth,1)+1:length(guiData.plot.psthLines{pltIdx}));


    newYLim = cell2mat(cellfun(@(x) [min([x.YData]) max([x.YData])], guiData.plot.psthLines, 'uni', 0));
    ylim(get(guiData.plot.psthLines{pltIdx}(1),'Parent'),[min(newYLim(:,1)), max(newYLim(:,2))]);

    newXLim = cell2mat(cellfun(@(x) [min([x.XData]) max([x.XData])], guiData.plot.psthLines, 'uni', 0));
    xlim(get(guiData.plot.psthLines{pltIdx}(1),'Parent'),[min(newXLim(:,1)), max(newXLim(:,2))]);

    % Plot raster
    % (single cluster mode)
    [~,~,rowGroup] = unique(currGroup,'sorted');
    if ~isempty(currAddTicks) && any(currAddTicks)
        [~, rasterSortIdx] = sortrows([currGroup, currAddTicks], [1,2]);
        currAddTicks = currAddTicks(rasterSortIdx);
    else
        [~, rasterSortIdx] = sortrows(currGroup);
    end
    currRaster = currRaster(rasterSortIdx,:);
    rowGroup = rowGroup(rasterSortIdx);

    [rasterY,rasterX] = find(currRaster);
    set(guiData.plot.rasterDots{pltIdx},'XData',guiData.plot.rasterTime(rasterX),'YData',rasterY);
    set(guiData.plot.rasterDots{pltIdx}, 'SizeData', 200/sqrt(size(currRaster,1)));
    ylim(get(guiData.plot.rasterDots{pltIdx},'Parent'),[0,size(tPeriEvent,1)]);
    if ~isempty(currAddTicks) && any(currAddTicks)
        set(guiData.plot.addRasterTicks{pltIdx},'XData',currAddTicks,'YData',1:length(currAddTicks));
    else
        set(guiData.plot.addRasterTicks{pltIdx},'XData',[nan nan],'YData',[nan nan]);
    end

    % (set dot color by group)
    psthColors = get(guiData.plot.psthLines{pltIdx},'color');
    if iscell(psthColors); psthColors = cell2mat(psthColors); end
    rasterDotColor = psthColors(rowGroup(rasterY),:);
    set(guiData.plot.rasterDots{pltIdx},'CData',rasterDotColor);


    % Plot cluster amplitude over whole experiment
    set(guiData.plot.amplitudes{pltIdx},'XData', ...
        guiData.curr.spkTimes{i}(currSpkIdx), ...
        'YData',guiData.curr.spkAmps{i}(currSpkIdx),'linestyle','none');
    
    assignin('base','guiData',guiData)
    set(guiData.titleSub{i}, 'String', sprintf('Alignment--%s   Grouping--%s', ...
        guiData.curr.evNames{i}, guiData.curr.grpNames{i}), 'visible', 1);
    addString = get(guiData.titleSub{i}, 'string');
end
if guiData.curr.nExps ~= 1
    addString = '';
    set(guiData.titleMain, 'Color', [0,0,0]);
else
    cellfun(@(x) set(x, 'visible', 0), guiData.titleSub, 'uni', 0);
    set(guiData.titleMain, 'Color', [0.8500 0.3250 0.0980]);
end
set(guiData.titleMain, 'String', sprintf('%s--%s   %s    Cluster--%d   Probe--%d of %d \n %s', ...
    strjoin(guiData.curr.subject, '&'), strjoin(guiData.curr.expDate, '&'), strjoin(guiData.curr.expDef, '&'), ...
    guiData.curr.cluster, guiData.curr.probe, guiData.curr.nProbes, addString));
guiData.titleMain_2 = 1;
end


function keyPress(cellrasterGui,eventdata)
% Get guidata
guiData = guidata(cellrasterGui);

switch eventdata.Key
    case 'p' %Switch probe
        if contains(eventdata.Modifier, 'shift')
            guiData.curr.probe = max([1 guiData.curr.probe - 1]);
        else
            guiData.curr.probe = min([guiData.curr.probe + 1, guiData.curr.nProbes]);
        end

    case 'd' %Switch date
        if contains(eventdata.Modifier, 'shift')
            guiData.curr.dIdx = max([1 guiData.curr.dIdx - 1]);
        else
            guiData.curr.dIdx = min([guiData.curr.dIdx + 1, guiData.nSessions]);
        end

    case 'downarrow' % Next cluster
        currClusterPosition = guiData.curr.cluster == guiData.curr.sortedTemplateIDs;
        newCluster = guiData.curr.sortedTemplateIDs(circshift(currClusterPosition,1));
        guiData.curr.cluster = newCluster;

    case 'uparrow' % Previous cluster
        currClusterPosition = guiData.curr.cluster(end) == guiData.curr.sortedTemplateIDs;
        newCluster = guiData.curr.sortedTemplateIDs(circshift(currClusterPosition,-1));
        guiData.curr.cluster = newCluster;

    case 'rightarrow' % Next trial group or event times (if shift pressed)
        if contains(eventdata.Modifier, 'shift')
            guiData.curr.evTimeRef = guiData.curr.evTimeRef + 1;
        else
            guiData.curr.triGrpRef = guiData.curr.triGrpRef + 1;
        end

    case 'leftarrow' % Previous trial group or event times (if shift pressed)
        if contains(eventdata.Modifier, 'shift')
            guiData.curr.evTimeRef = guiData.curr.evTimeRef - 1;
            guiData.sigRes = cellfun(@(x,y) neural.findResponsiveCells(x,y), guiData.curr.spks, guiData.curr.evTimes, 'uni', 0);
        else
            guiData.curr.triGrpRef = guiData.curr.triGrpRef - 1;
        end

    case 'c'
        % Enter and go to cluster
        newCluster = str2double(cell2mat(inputdlg('Go to cluster:')));
        if ~ismember(newCluster,unique(guiData.curr.sortedTemplateIDs))
            error(['Cluster ' num2str(newCluster) ' not present'])
        end
        guiData.curr.cluster = newCluster;
end

% Upload gui data and draw
if contains(eventdata.Key, {'p'; 'c'; 'arrow';'d'})
    assignGUIFields(cellrasterGui, guiData);
    if any(strcmpi(eventdata.Key, {'p';'d'})) || contains(eventdata.Modifier, 'shift')
        cycleProbe(cellrasterGui);
    end
    updatePlot(cellrasterGui);
end
end

function clusterClick(cellrasterGui,eventdata)
% Get guidata
guiData = guidata(cellrasterGui);

% Get the clicked cluster, update current cluster
clusterX = get(guiData.plot.clusterDots,'XData');
clusterY = get(guiData.plot.clusterDots,'YData');

%because yRange is much larger
ratio = (range(ylim)/range(xlim))/4;
ratioDivide = repmat([1; ratio], 1, length(clusterX)); 

clustDistance = [[clusterX;clusterY] - eventdata.IntersectionPoint(1:2)']./ratioDivide;
[~,clickedCluster] = min(sqrt(sum((clustDistance).^2,1)));
guiData.curr.cluster = guiData.curr.clusterIDs(clickedCluster);

% Upload gui data and draw
assignGUIFields(cellrasterGui, guiData);
updatePlot(cellrasterGui);
end

function matchYLim(ax1, ax2)
yLim1 = get(ax1, 'YLim');
set(ax2, 'YLim', yLim1);
end
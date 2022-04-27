function cellRaster(spk,eventTimes,trialGroups, opt)
%INPUTS
%spks with cluster IDs!!

% Raster viewer for spk data
%
% eventTimes - vector or cell array of times
% trialGroups - categorical vectors of trial types (optional)

% Controls: 

guiData = struct;
guiData.spk = spk;
guiData.spk2ClusterIdx = cellfun(@(x) arrayfun(@(y) x.spikes.cluster==y, [x.clusters.ID], 'uni', 0), spk, 'uni', 0);

% Validate eventTimes input
if ~exist('eventTimes','var'); error('No event times'); end
if ~exist('opt','var'); opt = struct; end
if ~iscell(eventTimes); eventTimes = {eventTimes}; end


% Validate trialGroups input
if ~exist('trialGroups','var') || isempty(trialGroups)
    trialGroups = cellfun(@(x) x*0+1,eventTimes,'uni',0); 
end
if ~iscell(trialGroups); trialGroups = {trialGroups}; end
if length(eventTimes) > 1 && length(trialGroups) < length(eventTimes)
    error('Mismatching align time/group sets'); 
end
missingUniformGroup = cellfun(@(x) ~all(x(:,1)==1), trialGroups);
trialGroups(missingUniformGroup) = cellfun(@(x) [x(:,1)*0+1, x],trialGroups(missingUniformGroup),'uni',0);

sizeCheck = cellfun(@(x,y) size(x,1)==size(y,1),eventTimes,trialGroups,'uni',0);
if ~all(cell2mat(sizeCheck))
    error('eventTimes & trialGroups should have same number of rows')
end

% Remove any NaN values from the event times and trial groups
trialGroups = cellfun(@(x,y) x(~isnan(y(:,1)),:), trialGroups, eventTimes, 'uni', 0);
eventTimes = cellfun(@(x) x(~isnan(x(:,1)),:), eventTimes, 'uni', 0);

guiData.eventTimes = eventTimes;
guiData.trialGroups = trialGroups;

%% Validate "opt"
guiData.clustNum = cellfun(@(x) length(x.clusters), spk);

if ~isfield(opt, 'highlight'); guiData.highlight = []; 
else, guiData.highlight = opt.highlight;
end
if ~isfield(opt,'sortRule'); guiData.sortRule = 'sig'; guiData.highlight = [];
else, guiData.sortRule = opt.sortRule;
end


%% Package gui data
nCol = 4;
nRow = 5;
nAxes = nCol*nRow;

cellrasterGui = figure('color','w');
guiData.unitAxes = subplot(nRow,nCol,1:nCol:(nAxes-nRow),'YDir','normal'); hold on;
xlim([-0.1,1]);
ylabel('Distance from tip (\mum)')
xlabel('xPosition (\mum)')
tRef = nCol-2:nCol;
disableDefaultInteractivity(gca)

guiData.psthAxes = subplot(nRow,nCol,tRef,'YAxisLocation','right'); hold on;
xlabel('Time from event (s)');
ylabel('spks/s/trial');
disableDefaultInteractivity(gca)

guiData.rasterAxes = subplot(nRow,nCol,[tRef+nCol,tRef+nCol*2,tRef+nCol*3], ...
    'YDir','reverse','YAxisLocation','right'); hold on;
xlabel('Time from event (s)');
ylabel('Trial');
disableDefaultInteractivity(gca)

guiData.amplitudeAxes = subplot(nRow,nCol,nAxes-nCol+1:nAxes); hold on;
xlabel('Time (s)');
ylabel('Amplitude');
axis tight
disableDefaultInteractivity(gca)

guiData.curr.eventTimeRef = 1;
guiData.curr.trialGroupRef = 1;
guiData.curr.probe = 1;
guiData.curr.totalProbes = length(guiData.spk);
guiData.title = annotation('textbox', [0.25, 0.98, 0.5, 0], 'string', 'My Text', 'EdgeColor', 'none',...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

guidata(cellrasterGui, guiData);
cycleProbe(cellrasterGui);
updatePlot(cellrasterGui);
end

function cycleProbe(cellrasterGui)
guiData = guidata(cellrasterGui);
spk = guiData.spk{guiData.curr.probe};

%SHOULD SET TITLE STRING HERE....
guiData.spkTimes = spk.spikes.time;
guiData.spkCluster = spk.spikes.cluster;
guiData.spkAmps = spk.spikes.tempScalingAmp;

guiData.clustDepths = [spk.clusters.Depth]';
guiData.clustXPos = [spk.clusters.XPos]';
guiData.clustIDs = [spk.clusters.ID]';

if min(guiData.spkCluster) == 0
    guiData.spkCluster = guiData.spkCluster+1; 
    guiData.clustIDs = guiData.clustIDs+1; 
end


guiData.curr.eventTimes = guiData.eventTimes{guiData.curr.eventTimeRef};
guiData.curr.trialGroups = guiData.trialGroups{guiData.curr.eventTimeRef};

switch guiData.sortRule(1:3)
    case 'dep'
        [~, guiData.curr.sortedClustIDs] = sort(guiData.clustDepths);
    case 'sig'
        guiData.sigRes = neural.findResponsiveCells(spk,guiData.curr.eventTimes);
        [~, guiData.curr.sortedClustIDs] = sort(guiData.sigRes.pVal);
        if isempty(guiData.highlight)
            guiData.highlight = guiData.sigRes.pVal<0.01;
        end
end
guiData.curr.sortedClustIDs = guiData.clustIDs(guiData.curr.sortedClustIDs);
guiData.curr.unit = guiData.curr.sortedClustIDs(1);
guiData.curr.unitIdx = find(guiData.curr.unit == guiData.clustIDs);


%%
% Initialize figure and axes
axes(guiData.unitAxes); cla;
ylim([min(guiData.clustDepths)-50 max(guiData.clustDepths)+50]);
xlim([min(guiData.clustXPos)-25 max(guiData.clustXPos)+25]);

% (plot unit depths by depth and relative number of spks)
if isempty(guiData.highlight); highlight = guiData.clustDepths*0;
else, highlight = guiData.highlight;
end
plot(guiData.clustXPos(highlight),guiData.clustDepths(highlight),'.b','MarkerSize',25);
unitDots = plot(guiData.clustXPos,guiData.clustDepths,'.k','MarkerSize',15,'ButtonDownFcn',@unitClick);

currUnitDot = plot(0,0,'.r','MarkerSize',20);

% (smoothed psth)
axes(guiData.psthAxes); cla;
maxNumGroups = 10; %HARDCODED for now... unlikely to be more than 10 trial labels...
psthLines = arrayfun(@(x) plot(NaN,NaN,'linewidth',2,'color','k'),1:maxNumGroups);

% (raster)
axes(guiData.rasterAxes); cla;
rasterDots = scatter(NaN,NaN,5,'k','filled');

% (spk amplitude across the recording)
axes(guiData.amplitudeAxes); cla;
amplitudePlot = plot(NaN,NaN,'.k');
amplitudeLines = arrayfun(@(x) line([0,0],ylim,'linewidth',2),1:2);

% Set default raster times
rasterWindow = [-0.5,1];
psthBinSize = 0.001;
tBins = rasterWindow(1):psthBinSize:rasterWindow(2);
t = tBins(1:end-1) + diff(tBins)./2;

[~, trialGroupSort] = sort(guiData.curr.trialGroups(:, guiData.curr.trialGroupRef));
tPeriEvent = guiData.curr.eventTimes(trialGroupSort) + tBins;
% (handle NaNs by setting rows with NaN times to 0)
tPeriEvent(any(isnan(tPeriEvent),2),:) = 0;

% Set functions for key presses
set(cellrasterGui,'KeyPressFcn',@keyPress);

% (plots)
guiData.unitDots = unitDots;
guiData.currUnitDot = currUnitDot;
guiData.psthLines = psthLines;
guiData.rasterDots = rasterDots;
guiData.amplitudePlot = amplitudePlot;
guiData.amplitudeLines = amplitudeLines; 

% (raster times)
guiData.t = t;
guiData.tBins = tBins;
guiData.tPeriEvent = tPeriEvent;

% Upload gui data and draw
guidata(cellrasterGui, guiData);
end

function updatePlot(cellrasterGui)
% Get guidata
guiData = guidata(cellrasterGui);
guiData.curr.unitIdx = find(guiData.curr.unit == guiData.clustIDs);

% Turn on/off the appropriate graphics
set(guiData.rasterDots,'visible','on');

% Plot depth location on probe
unitX = get(guiData.unitDots,'XData');
unitY = get(guiData.unitDots,'YData');
set(guiData.currUnitDot,'XData',unitX(guiData.curr.unitIdx), 'YData',unitY(guiData.curr.unitIdx));

% Bin spks (use only spks within time range, big speed-up)
currspkIdx = ismember(guiData.spkCluster,guiData.curr.unit);
currRasterspkTimes = guiData.spkTimes(currspkIdx);
currRasterspkTimes(currRasterspkTimes < min(guiData.tPeriEvent(:)) | ...
    currRasterspkTimes > max(guiData.tPeriEvent(:))) = [];

tDat = guiData.tPeriEvent';
if ~any(diff(tDat(:))<0)
    currRaster = [histcounts(currRasterspkTimes,tDat(:)),0];
    currRaster = reshape(currRaster, size(guiData.tPeriEvent'))';
    currRaster(:,end) = [];
else
    currRaster = cell2mat(arrayfun(@(x) ...
        histcounts(currRasterspkTimes,guiData.tPeriEvent(x,:)), ...
        [1:size(guiData.tPeriEvent,1)]','uni',false));
end

% Set color scheme
currGroup = guiData.curr.trialGroups(:,guiData.curr.trialGroupRef);
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
gw = gausswin(51,3)';
smWin = gw./sum(gw);
currPsth = grpstats(currRaster,currGroup,@(x) mean(x,1));
padMatrix = zeros(size(currPsth,1), floor(length(smWin)/2));
currSmoothedPsth = conv2([padMatrix,currPsth,padMatrix], smWin,'valid')./mean(diff(guiData.t));

% (set the first n lines by group, set all others to NaN)
arrayfun(@(x) set(guiData.psthLines(x), ...
    'XData',guiData.t,'YData',currSmoothedPsth(x,:), ...
    'Color',groupColors(x,:)),1:size(currPsth,1));
arrayfun(@(align_group) set(guiData.psthLines(align_group), ...
    'XData',NaN,'YData',NaN), ...
    size(currPsth,1)+1:length(guiData.psthLines));

ylim(get(guiData.psthLines(1),'Parent'),[min(currSmoothedPsth(:)), ...
    max(max(currSmoothedPsth(:),min(currSmoothedPsth(:))+1))]);

% Plot raster
% (single unit mode)
[~,~,rowGroup] = unique(guiData.curr.trialGroups(:,guiData.curr.trialGroupRef),'sorted');
[~, rasterSortIdx] = sort(currGroup);
currRaster = currRaster(rasterSortIdx,:);
rowGroup = rowGroup(rasterSortIdx);

[rasterY,rasterX] = find(currRaster);
set(guiData.rasterDots,'XData',guiData.t(rasterX),'YData',rasterY);
xlim(get(guiData.rasterDots,'Parent'),[guiData.tBins(1),guiData.tBins(end)]);
ylim(get(guiData.rasterDots,'Parent'),[0,size(guiData.tPeriEvent,1)]);
% (set dot color by group)
psthColors = get(guiData.psthLines,'color');
if iscell(psthColors); psthColors = cell2mat(psthColors); end
rasterDotColor = psthColors(rowGroup(rasterY),:);
set(guiData.rasterDots,'CData',rasterDotColor);


% Plot template amplitude over whole experiment
set(guiData.amplitudePlot,'XData', ...
    guiData.spkTimes(currspkIdx), ...
    'YData',guiData.spkAmps(currspkIdx),'linestyle','none');

[ymin,ymax] = bounds(get(guiData.amplitudePlot,'YData'));
set(guiData.amplitudeLines(1),'XData',repmat(min(guiData.tPeriEvent(:)),2,1),'YData',[ymin,ymax]);
set(guiData.amplitudeLines(2),'XData',repmat(max(guiData.tPeriEvent(:)),2,1),'YData',[ymin,ymax]);
assignin('base','guiData',guiData)

set(guiData.title, 'String', sprintf('ClusterID: %d', guiData.curr.unit));
end


function keyPress(cellrasterGui,eventdata)
% Get guidata
guiData = guidata(cellrasterGui);

switch eventdata.Key
    case {'p'} %Switch probe
        guiData.curr.probe = guiData.curr.probe + 1;
        if guiData.curr.probe > guiData.curr.totalProbes
            guiData.curr.probe = 1;
        end
        guidata(cellrasterGui,guiData);
        cycleProbe(cellrasterGui);
        guiData = guidata(cellrasterGui);
        
    case 'downarrow'
        % Next unit
        currUnitPosition = guiData.curr.unit == guiData.curr.sortedClustIDs;
        newUnit = guiData.curr.sortedClustIDs(circshift(currUnitPosition,1));
        guiData.curr.unit = newUnit;
        
    case 'uparrow'
        % Previous unit
        currUnitPosition = guiData.curr.unit(end) == guiData.curr.sortedClustIDs;
        newUnit = guiData.curr.sortedClustIDs(circshift(currUnitPosition,-1));
        guiData.curr.unit = newUnit;

    case 'rightarrow'
        % Next group
        newTrialGroup = guiData.curr.trialGroupRef + 1;
        if newTrialGroup > size(guiData.trialGroups{guiData.curr.eventTimeRef},2)
            newTrialGroup = 1;
        end
        guiData.curr.trialGroupRef = newTrialGroup;
        
    case 'leftarrow'
        % Previous group
        newTrialGroup = guiData.curr.trialGroupRef - 1;
        if newTrialGroup < 1
            newTrialGroup = size(guiData.trialGroups{guiData.curr.eventTimeRef},2);
        end
        guiData.curr.trialGroupRef = newTrialGroup; 

        
%     case 'pagedown'
%         % Next alignment
%         newAlign = guiData.currAlign + 1;
%         if newAlign > length(guiData.currEventTimes)
%             newAlign = 1;
%         end
%         useAlign = reshape(guiData.currEventTimes{newAlign},[],1);
%         tPeriEvent = useAlign + guiData.tBins;
%         
%         % (handle NaNs by setting rows with NaN times to 0)
%         tPeriEvent(any(isnan(tPeriEvent),2),:) = 0;
%         
%         guiData.currAlign = newAlign;
%         guiData.tPeriEvent = tPeriEvent;
% %         guiData.currGroup = 1;
%         
%     case 'pageup'
%         % Previous alignment
% 
%         newAlign = guiData.currAlign - 1;
%         if newAlign < 1
%             newAlign = length(guiData.currEventTimes);
%         end
%         useAlign = reshape(guiData.currEventTimes{newAlign},[],1);
%         tPeriEvent = useAlign + guiData.tBins;
%         
%         % (handle NaNs by setting rows with NaN times to 0)
%         tPeriEvent(any(isnan(tPeriEvent),2),:) = 0;
%         
%         guiData.currAlign = newAlign;
%         guiData.tPeriEvent = tPeriEvent;
% %         guiData.currGroup = 1;
        
    case 'u'
        % Enter and go to unit
        newUnit = str2double(cell2mat(inputdlg('Go to unit:')));
        if ~ismember(newUnit,unique(guiData.curr.sortedClustIDs))
            error(['Unit ' num2str(newUnit) ' not present'])
        end
        guiData.curr.unit = newUnit;
        
end

% Upload gui data and draw
guidata(cellrasterGui,guiData);
updatePlot(cellrasterGui);
end

function unitClick(cellrasterGui,eventdata)

% Get guidata
guiData = guidata(cellrasterGui);

% Get the clicked unit, update current unit
unitX = get(guiData.unitDots,'XData');
unitY = get(guiData.unitDots,'YData');

[~,clickedUnit] = min(sqrt(sum(([unitX;unitY] - ...
    eventdata.IntersectionPoint(1:2)').^2,1)));
guiData.curr.unit = guiData.clustIDs(clickedUnit);

% Upload gui data and draw
guidata(cellrasterGui,guiData);
updatePlot(cellrasterGui);
end

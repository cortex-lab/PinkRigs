function cellRaster(blk,eventTimes,trialGroups,sortRule)
%INPUTS
%spks with cluster IDs!!


% AP_cellraster(eventTimes,trialGroups,sortRule)
%
% Raster viewer for Neuropixels
%
% eventTimes - vector or cell array of times
% trialGroups - categorical vectors of trial types (optional)
% sortRule - sorting for scrolling through units (default = depth)
%
% Controls: 
% up/down - switch between units (clicking on unit also selects)
% left/right - switch between alignments (if multiple)
% u - go to unit number
% s - change experiment number

guiData = struct;
guiData.blk = blk;

% Initiate eventTimes
% (align times required as input)
if ~exist('eventTimes','var'); error('No align times'); end
if exist('sortRule','var'); guiData.sortRule = sortRule; else; guiData.sortRule = 'sig'; end

% (put eventTimes into cell array if it isn't already, standardize dim)
if ~iscell(eventTimes); eventTimes = {eventTimes}; end
for i = find(cellfun(@(x) size(x,2)==1, eventTimes))'
    if length(eventTimes{i}) == length(blk.tri.expRef); eventTimes{i} = [eventTimes{i} double(blk.tri.expRef)];
    else, error('Need to know which experiments each event relates to');
    end
end

% Initiate trialGroups
% (if no align groups specified, create one group for each alignment)
if ~exist('trialGroups','var') || isempty(trialGroups); trialGroups =  cellfun(@(x) ones(size(x,1),1),eventTimes,'uni',false); end
if ~iscell(trialGroups); trialGroups = {trialGroups}; end
if length(eventTimes) > 1 && length(trialGroups) < length(eventTimes); error('Mismatching align time/group sets'); end

% (check group against time dimensions, orient align times x groups)
groupDim = cellfun(@(align,group) find(ismember(size(group),length(align))),eventTimes,trialGroups,'uni',false);
if any(cellfun(@isempty,groupDim))
    error('Mismatching times/groups within align set')
end
trialGroups = cellfun(@(groups,dim) shiftdim(groups,dim-1),trialGroups,groupDim,'uni',false);

trialGroups = cellfun(@(x,y) x(~isnan(y(:,1)),:), trialGroups, eventTimes, 'uni', 0);
eventTimes = cellfun(@(x) x(~isnan(x(:,1)),:), eventTimes, 'uni', 0);

% (if there isn't an all ones category first, make one)
guiData.trialGroups = cellfun(@(x) padarray(x,[0,1-all(x(:,1) == 1)],1,'pre'),trialGroups,'uni',false);

% Package gui data
cellrasterGui = figure('color','w');
guiData.penetrationIdxs = unique(guiData.blk.pen.ephysRecordIdx);
guiData.currPenetration = guiData.penetrationIdxs(1);
guiData.eventTimes = eventTimes;

guiData.unitAxes = subplot(5,5,1:5:20,'YDir','reverse'); hold on;
xlim([-0.1,1]);
ylabel('Depth (\mum)')
xlabel('Normalized log rate')

guiData.waveformAxes = subplot(5,5,2:5:20,'visible','off'); hold on;
linkaxes([guiData.unitAxes,guiData.waveformAxes],'y');

guiData.psthAxes = subplot(5,5,[3,4,5],'YAxisLocation','right'); hold on;
xlabel('Time from event (s)');
ylabel('spks/s/trial');

guiData.rasterAxes = subplot(5,5,[8,9,10,13,14,15,18,19,20],'YDir','reverse','YAxisLocation','right'); hold on;
xlabel('Time from event (s)');
ylabel('Trial');

guiData.amplitudeAxes = subplot(5,5,21:25); hold on;
xlabel('Experiment time (s)');
ylabel('Template amplitude');
axis tight

guiData.currAlign = 1;
guiData.currGroup = 1;
guiData.title = annotation('textbox', [0.25, 0.98, 0.5, 0], 'string', 'My Text', 'EdgeColor', 'none',...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');

guidata(cellrasterGui, guiData);
cyclePenetration(cellrasterGui);
updatePlot(cellrasterGui);
end

function cyclePenetration(cellrasterGui)
guiData = guidata(cellrasterGui);
blk = guiData.blk;
blk = prc.filtBlock(blk, guiData.currPenetration==blk.pen.ephysRecordIdx, 'penetration');
if ~isfield(blk.clu, 'highlight'); blk.clu.highlight = (blk.clu.depths*0)==0; end

set(guiData.title, 'String', sprintf('%s: %s Penetration %d', blk.exp.subject{1},blk.exp.expDate{1},guiData.currPenetration));
channelPos = blk.pen.channelMap{1};
top20 = blk.clu.templates;
cluTemplates = zeros(size(top20, 1), size(top20,2)-1,size(channelPos,1));
for i = 1:size(top20, 1)
     cluTemplates(i,:,top20(i,end,:)) = top20(i,1:end-1,:);
end
expRefIdx = guiData.blk.pen.expRef(guiData.currPenetration==guiData.blk.pen.ephysRecordIdx);

guiData.channelPositions = channelPos;
guiData.clusterTemplates = cluTemplates;
guiData.spkTimes = cell2mat(blk.clu.spkTimes);
guiData.spkCluster = cell2mat(cellfun(@(x,y) x*0+y, blk.clu.spkTimes, num2cell((1:length(blk.clu.depths))'), 'uni', 0));
guiData.spkAmps = cell2mat(blk.clu.spkAmplitudes);

guiData.currEventTimes = cellfun(@(x) x(x(:,2)==expRefIdx,1), guiData.eventTimes, 'uni', 0);
guiData.currTrialGroups = cellfun(@(x,y) x(y(:,2)==expRefIdx,:), guiData.trialGroups, guiData.eventTimes, 'uni', 0);

switch guiData.sortRule(1:3)
    case 'dep'
        [~, guiData.currSortRule] = sort(blk.clu.depths);
    case 'sig'
        guiData.clusterSigLevel = cellfun(@(x) kil.findResponsiveCells(blk,x), [guiData.currEventTimes guiData.currEventTimes{1}*0+1], 'uni', 0);
        [~, guiData.currSortRule] = sort(min(cell2mat(guiData.clusterSigLevel'),[],2));
end
guiData.currEventTimes = guiData.currEventTimes(:,1);
guiData.currUnit = guiData.currSortRule(1);


% Initialize figure and axes
axes(guiData.unitAxes); cla;
ylim([-50, max(guiData.channelPositions(:,2))+50]);

% (plot unit depths by depth and relative number of spks)
normspkNum = mat2gray(log(accumarray(guiData.spkCluster,1)+1));
if length(normspkNum)<length(blk.clu.depths); normspkNum(end+1:length(blk.clu.depths)) = min(normspkNum); end
highlight = blk.clu.highlight;
plot(normspkNum(highlight),blk.clu.depths(highlight),'.b','MarkerSize',25);
unitDots = plot(normspkNum,blk.clu.depths,'.k','MarkerSize',15,'ButtonDownFcn',@unitClick);

currUnitDots = plot(0,0,'.r','MarkerSize',20);

% (plot of waveform across the probe)
axes(guiData.waveformAxes); cla;
ylim([-50, max(guiData.channelPositions(:,2))+50]);
waveformLines = arrayfun(@(x) plot(guiData.waveformAxes,0,0,'k','linewidth',1),1:size(guiData.clusterTemplates,3));

% (smoothed psth)
axes(guiData.psthAxes); cla;
maxNumGroups = max(max(cell2mat(cellfun(@(x) 1+sum(diff(sort(x,1),[],1) ~= 0),guiData.currTrialGroups,'uni',0))));
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
useAlign = reshape(guiData.currEventTimes{guiData.currAlign},[],1);
tPeriEvent = useAlign + tBins;
% (handle NaNs by setting rows with NaN times to 0)
tPeriEvent(any(isnan(tPeriEvent),2),:) = 0;

% Set functions for key presses
set(cellrasterGui,'KeyPressFcn',@keyPress);

% (plots)
guiData.unitDots = unitDots;
guiData.currUnitDots = currUnitDots;
guiData.waveformLines = waveformLines;
guiData.psthLines = psthLines;
guiData.rasterDots = rasterDots;
guiData.amplitudePlot = amplitudePlot;
guiData.amplitudeLines = amplitudeLines; 

% (raster times)
guiData.t = t;
guiData.tBins = tBins;
guiData.tPeriEvent = tPeriEvent;

% (spk data)


% Upload gui data and draw
guidata(cellrasterGui, guiData);
end


function updatePlot(cellrasterGui)
% Get guidata
guiData = guidata(cellrasterGui);

% Turn on/off the appropriate graphics
set(guiData.rasterDots,'visible','on');

% Plot depth location on probe
unitX = get(guiData.unitDots,'XData');
unitY = get(guiData.unitDots,'YData');
set(guiData.currUnitDots,'XData',unitX(guiData.currUnit), 'YData',unitY(guiData.currUnit));

% Plot waveform across probe (reversed YDir, flip Y axis and plot depth)
templateXScale = 7;
templateYScale = 250;

templateY = permute(mean(guiData.clusterTemplates(guiData.currUnit,:,:),1),[3,2,1]);
templateY = -templateY*templateYScale + guiData.channelPositions(:,2);
templateX = (1:size(guiData.clusterTemplates,2)) + guiData.channelPositions(:,1)*templateXScale;

templateChannelAmp = range(guiData.clusterTemplates(guiData.currUnit,:,:),2);
templateThresh = max(templateChannelAmp,[],3)*0.2;
templateUseChannels = any(templateChannelAmp > templateThresh,1);
[~,maxChannel] = max(max(abs(guiData.clusterTemplates(guiData.currUnit,:,:)),[],2),[],3);

arrayfun(@(ch) set(guiData.waveformLines(ch),'XData',templateX(ch,:),'YData',templateY(ch,:)),1:size(guiData.clusterTemplates,3));
arrayfun(@(ch) set(guiData.waveformLines(ch),'Color','r'),find(templateUseChannels));
arrayfun(@(ch) set(guiData.waveformLines(ch),'Color','k'),find(~templateUseChannels));
set(guiData.waveformLines(maxChannel),'Color','b');

% Bin spks (use only spks within time range, big speed-up)
currspkIdx = ismember(guiData.spkCluster,guiData.currUnit);
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
currGroup = guiData.currTrialGroups{guiData.currAlign}(:,guiData.currGroup);
if length(unique(currGroup)) == 1
    % Black if one group
    group_colors = [0,0,0];
elseif length(unique(sign(currGroup(currGroup ~= 0)))) == 1
    % Black-to-red single-signed groups
    n_groups = length(unique(currGroup));
    group_colors = [linspace(0,0.8,n_groups)',zeros(n_groups,1),zeros(n_groups,1)];
elseif length(unique(sign(currGroup(currGroup ~= 0)))) == 2
    % Symmetrical blue-black-red if negative and positive groups
    n_groups_pos = length(unique(currGroup(currGroup > 0)));
    group_colors_pos = [linspace(0.3,1,n_groups_pos)',zeros(n_groups_pos,1),zeros(n_groups_pos,1)];
    
    n_groups_neg = length(unique(currGroup(currGroup < 0)));
    group_colors_neg = [zeros(n_groups_neg,1),zeros(n_groups_neg,1),linspace(0.3,1,n_groups_neg)'];
    
    n_groups_zero = length(unique(currGroup(currGroup == 0)));
    group_colors_zero = [zeros(n_groups_zero,1),zeros(n_groups_zero,1),zeros(n_groups_zero,1)];
    
    group_colors = [flipud(group_colors_neg);group_colors_zero;group_colors_pos];    
end

% Plot smoothed PSTH
gw = gausswin(51,3)';
smWin = gw./sum(gw);
currPsth = grpstats(currRaster,currGroup,@(x) mean(x,1));
currSmoothedPsth = conv2(padarray(currPsth, ...
    [0,floor(length(smWin)/2)],'replicate','both'), ...
    smWin,'valid')./mean(diff(guiData.t));

% (set the first n lines by group, set all others to NaN)
arrayfun(@(align_group) set(guiData.psthLines(align_group), ...
    'XData',guiData.t,'YData',currSmoothedPsth(align_group,:), ...
    'Color',group_colors(align_group,:)),1:size(currPsth,1));
arrayfun(@(align_group) set(guiData.psthLines(align_group), ...
    'XData',NaN,'YData',NaN), ...
    size(currPsth,1)+1:length(guiData.psthLines));

ylim(get(guiData.psthLines(1),'Parent'),[min(currSmoothedPsth(:)), ...
    max(max(currSmoothedPsth(:),min(currSmoothedPsth(:))+1))]);
title(get(guiData.psthLines(1),'Parent'), ...
    ['Unit ' num2str(guiData.currUnit) ...
    ', Align ' num2str(guiData.currAlign) ...
    ', Group ' num2str(guiData.currGroup)],'FontSize',14);


% Plot raster
% (single unit mode)
[~,~,rowGroup] = unique(guiData.currTrialGroups{guiData.currAlign}(:,guiData.currGroup),'sorted');
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
end


function keyPress(cellrasterGui,eventdata)

% Get guidata
guiData = guidata(cellrasterGui);

switch eventdata.Key
    case {'equal';'hyphen';'s'}
        if contains(eventdata.Key, {'equal';'s'}); moveDir = 1; else, moveDir = -1; end
        guiData.penetrationIdxs = circshift(guiData.penetrationIdxs, [-moveDir, 0]);
        guiData.currPenetration = guiData.penetrationIdxs(1);
        guidata(cellrasterGui,guiData);
        cyclePenetration(cellrasterGui);
        guiData = guidata(cellrasterGui);
        
    case 'downarrow'
        % Next unit
        currUnitIdx = guiData.currUnit(1) == guiData.currSortRule;
        newUnit = guiData.currSortRule(circshift(currUnitIdx,1));
        guiData.currUnit = newUnit;
        
    case 'uparrow'
        % Previous unit
        currUnitIdx = guiData.currUnit(end) == guiData.currSortRule;
        newUnit = guiData.currSortRule(circshift(currUnitIdx,-1));
        guiData.currUnit = newUnit;
        
    case 'rightarrow'
        % Next alignment
        newAlign = guiData.currAlign + 1;
        if newAlign > length(guiData.currEventTimes)
            newAlign = 1;
        end
        useAlign = reshape(guiData.currEventTimes{newAlign},[],1);
        tPeriEvent = useAlign + guiData.tBins;
        
        % (handle NaNs by setting rows with NaN times to 0)
        tPeriEvent(any(isnan(tPeriEvent),2),:) = 0;
        
        guiData.currAlign = newAlign;
        guiData.tPeriEvent = tPeriEvent;
%         guiData.currGroup = 1;
        
    case 'leftarrow'
        % Previous alignment
        newAlign = guiData.currAlign - 1;
        if newAlign < 1
            newAlign = length(guiData.currEventTimes);
        end
        useAlign = reshape(guiData.currEventTimes{newAlign},[],1);
        tPeriEvent = useAlign + guiData.tBins;
        
        % (handle NaNs by setting rows with NaN times to 0)
        tPeriEvent(any(isnan(tPeriEvent),2),:) = 0;
        
        guiData.currAlign = newAlign;
        guiData.tPeriEvent = tPeriEvent;
%         guiData.currGroup = 1;

    case 'pagedown'
        % Next group
        nextGroup = guiData.currGroup + 1;
        if nextGroup > size(guiData.currTrialGroups{guiData.currAlign},2)
            nextGroup = 1;
        end
        guiData.currGroup = nextGroup;
        
    case 'pageup'
        % Previous group
        nextGroup = guiData.currGroup - 1;
        if nextGroup < 1
            nextGroup = size(guiData.currTrialGroups{guiData.currAlign},2);
        end
        guiData.currGroup = nextGroup;     
        
    case 'u'
        % Enter and go to unit
        newUnit = str2num(cell2mat(inputdlg('Go to unit:')));
        if ~ismember(newUnit,unique(guiData.spkCluster))
            error(['Unit ' num2str(newUnit) ' not present'])
        end
        guiData.currUnit = newUnit;
        
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

[~,clicked_unit] = min(sqrt(sum(([unitX;unitY] - ...
    eventdata.IntersectionPoint(1:2)').^2,1)));
guiData.currUnit = clicked_unit;

% Upload gui data and draw
guidata(cellrasterGui,guiData);
updatePlot(cellrasterGui);
end

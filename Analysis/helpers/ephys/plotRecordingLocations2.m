% clear all
% close all

%% Get experiments

clear params
params.mice2Check = 'AV007';
params.days2Check = inf;
params.expDef2Check = 'multiSpaceWorld_checker_training';
params.preproc2Check = '1,*';
exp2checkList = csv.queryExp(params); 
pltClusters = 1;

%% Get the probes layout

% Get probes layout
mainCSV = csv.readTable(csv.getLocation('main'));
mouseIdx = strcmpi(mainCSV.Subject,params.mice2Check);
probeNum = ~isempty(mainCSV(mouseIdx,:).P0_type) + ~isempty(mainCSV(mouseIdx,:).P1_type);

% Get the basic layout
chanPosAll = [];
chanProbeSerialNoAll = [];
shankNum = zeros(probeNum,1);
elecPos = cell(probeNum,1);
probeType = cell(probeNum,1);
probeSerialNo = nan(probeNum,1);
for pp = 1:probeNum
    probeType{pp} = mainCSV(mouseIdx,:).(sprintf('P%d_type',pp-1)){1};
    probeSerialNo(pp) = str2double(mainCSV(mouseIdx,:).(sprintf('P%d_serialNo',pp-1)){1});
    switch probeType{pp}
        case '2.0 - 4shank'
            % Taken from the IMRO plotting file
            shankNum(pp) = 4;
            nElec = 1280;   % per shank; pattern repeats for the four shanks
            vSep = 15;      % in um
            hSep = 32;
            shankSep = 200; % That's what's in the ephys metadata...
            
            % elecPos is the electrode position within one shank
            elecPos{pp} = zeros(nElec, 2);
            elecPos{pp}(1:2:end,1) = 0;                %sites 0,2,4...
            elecPos{pp}(2:2:end,1) =  hSep;            %sites 1,3,5...
            % fill in y values
            viHalf = (0:(nElec/2-1))';                %row numbers
            elecPos{pp}(1:2:end,2) = viHalf * vSep;       %sites 0,2,4...
            elecPos{pp}(2:2:end,2) = elecPos{pp}(1:2:end,2);  %sites 1,3,5...
        otherwise
            error('Not yet coded.')
    end
    
    % Get all channels' position
    for shank = 1:shankNum(pp)
        elecPosPlt = nan(size(elecPos{pp}));
        elecPosPlt(:,1) = shankSep*(shank-1) + elecPos{pp}(:,1);
        elecPosPlt(:,2) = elecPos{pp}(:,2);
        
        chanPosAll = [chanPosAll; elecPosPlt];
        chanProbeSerialNoAll = [chanProbeSerialNoAll; ones(size(elecPosPlt,1),1)*probeSerialNo(pp)];
    end
end
[~,probePos] = max(chanProbeSerialNoAll == probeSerialNo',[],2);
probePos = (probePos-1)*sum(shankNum)*shankSep; %%% won't work if more than 2 probes (sum(shankNum))
chanPosAllPlt = [chanPosAll(:,1) + probePos, chanPosAll(:,2)]; % Easier if have two different things here -- actual position and plot position

%% Load all recording information for all channels

chanExpRef = cell(size(chanPosAll,1),1);
plotBehData = cell(size(exp2checkList,1),1);
for ee = 1:size(exp2checkList,1)
    
    % Get exp info
    expInfo = exp2checkList(ee,:);
    expPath = expInfo.expFolder{1};
    
    % Get which channels were recorded from
    % temporarily dead
    % ephysPaths = regexp(expInfo.ephysRecordingPath,',','split');
    % ephysPaths = ephysPaths{1};
    alignmentFile = dir(fullfile(expPath,'*alignment.mat'));
    load(fullfile(alignmentFile.folder, alignmentFile.name), 'ephys');
    ephysPaths = {ephys.ephysPath}; clear ephys
    
    for pp = 1:numel(ephysPaths)
        binFile = dir(fullfile(ephysPaths{pp},'*ap.bin'));
        if isempty(binFile)
            fprintf('No bin file in %s. Skipping.\n',ephysPaths{pp})
        else
            metaData = readMetaData_spikeGLX(binFile.name,binFile.folder);
            
            %% Extract info from metadata
                        
            % Which probe
            probeSerialNoExp = str2double(metaData.imDatPrb_sn);
            
            % Get recorded channel location
            % Same as in plotIMROProtocol
            out = regexp(metaData.imroTbl,'\(|\)(|\)','split');
            out(1:2) = []; % empty + extra channel or something?
            out(end) = []; % empty
            
            chans = nan(1,numel(out));
            shank = nan(1,numel(out));
            bank = nan(1,numel(out));
            elecInd = nan(1,numel(out));
            for c = 1:numel(out)
                chanProp = regexp(out{c},' ','split');
                chans(c) = str2double(chanProp{1});
                shank(c) = str2double(chanProp{2});
                bank(c) = str2double(chanProp{3});
                % 4 is refElec
                elecInd(c) = str2double(chanProp{5});
            end
            chanPosRec = elecPos{probeSerialNo == probeSerialNoExp}(elecInd+1,:);
            
            for sI = 0:3
                cc = find(shank == sI);
                chanPosRec(cc,1) = chanPosRec(cc,1) + shankSep*sI;
            end

            % Assign experiment ref to channels
            for c = 1:size(chanPos,1)
                chanIdx = all(chanPosAll == chanPosRec(c,:),2) & (chanProbeSerialNoAll == probeSerialNoExp);
                chanExpRef{chanIdx} = [chanExpRef{chanIdx} ee];
            end
        end
    end
    
    % Get behavior
    opt.noPlot = 1;
    opt.expNum = expInfo.expNum;
    plotBehData(ee) = plt.behaviour.boxPlots({params.mice2Check},expInfo.expDate, 'res', expInfo.expDef, opt);
end

%% Plot the basic layout

fig = figure('Position', [1 31 1920 973],'Name', params.mice2Check); 
clear probeAxes behaviorAxes

recNumPerChan = cellfun(@(x) size(x,2), chanExpRef);
nRow = ceil(max(recNumPerChan)/4);
nCol = 4;

% Set the layout for the probes
axesProbes = subplot(nRow,nCol,(0:nRow-1)*nCol+1); hold all
% scatter(chanPosAllPlt(:,1), chanPosAllPlt(:,2), 30, 'k', 'square')
trialNumChan = cellfun(@(x) sum(cellfun(@(y) y.totTrials, plotBehData(x))), chanExpRef);
imageProbes = nan(ceil(max(chanPosAllPlt(:,1))/15),ceil(max(chanPosAllPlt(:,2))/15)); % 15 * 15 um map
for c = 1:size(chanPosAllPlt,1)
    imageProbes(floor(chanPosAllPlt(c,1)/15)+1,floor(chanPosAllPlt(c,2)/15)+1) = trialNumChan(c);
end
imProbes = imagesc(0:15:max(chanPosAllPlt(:,1)),0:15:max(chanPosAllPlt(:,2)),imageProbes','AlphaData',~isnan(imageProbes'));
colormap(axesProbes,'parula')
cb = colorbar; 
cb.Position = [.27 .8 .01 .1];
cb.Title.String = 'Trial count';
xlim([-100 3000]);
ylim([0 6000]);

% Add the selected channel
elecSelectCoord = [0 0];
eleSecScat = scatter(elecSelectCoord(1), elecSelectCoord(2), 35, 'r', 'square','LineWidth',3);

% Add a reference recording to visualize clusters density
if pltClusters
    paramsClu.mice2Check = params.mice2Check;
    paramsClu.days2Check = {'2022-03-28','2022-03-29'}; % find a better way 
    paramsClu.expDef2Check = 'spontaneousActivity';
    exp2checkListClu = csv.queryExp(paramsClu);
    
    qMetricFilter = 2;
    if qMetricFilter == 1
        % get good units
        bc_qualityParamValues;
        paramBC = param; clear param;
        paramBC.somatic = [0 1];
        paramBC.minAmplitude = 10;
    end
    
    for ee = 1:size(exp2checkListClu,1)
        preprocFile = dir(fullfile(exp2checkListClu(ee,:).expFolder{1},'*preprocData.mat'));
        preprocDat = load(fullfile(preprocFile.folder,preprocFile.name),'spk');
        
        for pp = 1:numel(preprocDat.spk)
            if iscell(preprocDat.spk)
                if qMetricFilter == 1
                    % get good units
                    bc_qualityParamValues;
                    param.somatic = [0 1];
                    param.minAmplitude = 10;
                    
                    unitType = nan(1,numel(preprocDat.spk{pp}.clusters));
                    for c = 1:numel(preprocDat.spk{pp}.clusters)
                        unitType(c) = bc_getQualityUnitType(preprocDat.spk{pp}.clusters(c),paramBC);
                    end
                    goodUnits = unitType == 1;
                elseif qMetricFilter == 2
                    goodUnits = [preprocDat.spk{pp}.clusters.KSLab] == 2;
                else
                    goodUnits = true(1,numel(preprocDat.spk{pp}.clusters.KSLab));
                end
                
                scatter([preprocDat.spk{pp}.clusters(goodUnits).XPos] + (pp-1)*sum(shankNum)*shankSep + shankSep/3, ...
                    [preprocDat.spk{pp}.clusters(goodUnits).Depth],...
                    30,'k','filled')
            end
        end
    end
end

% Set the layout for behavior
ss = 0;
clear axesBehavior imBehavior textBehavior
for nr = 1:nRow
    for nc = 1:nCol-1
        ss = ss+1;
        axesBehavior(ss) = subplot(nRow,nCol,nCol*(nr-1) + nc +1); hold all
        axesBehavior(ss).Title.String = 'No session';
        imBehavior(ss) = imagesc(nan,'AlphaData',~isnan(nan));
        textBehavior{ss} =  text(1,1,'No trial','horizontalalignment', 'center','Parent',axesBehavior(ss));
    end
end

%% Interactive plot of behavior

keepGoing = 1;
while keepGoing
    was_a_key = waitforbuttonpress;
    if was_a_key && strcmp(get(fig, 'CurrentKey'), 'leftarrow')
        % change view
    elseif was_a_key && strcmp(get(fig, 'CurrentKey'), 'rightarrow')
        % change view
    elseif was_a_key && strcmp(get(fig, 'CurrentKey'), 'escape')
        keepGoing = 0;
    else
        cp = get(gca, 'CurrentPoint');
        [~,xidx] = min(abs(chanPosAllPlt(:,1)-cp(1,1)));
        xposPlt = chanPosAllPlt(xidx,1);
        [~,yidx] = min(abs(chanPosAllPlt(:,2)-cp(1,2)));
        yposPlt = chanPosAllPlt(yidx,2);
        
        % Reposition selected channel 
        eleSecScat.XData = xposPlt;
        eleSecScat.YData = yposPlt;
        
        eleSec = find(all(chanPosAllPlt == [xposPlt yposPlt],2));
        
        % Update associated behavior
        for ss = 1:numel(axesBehavior)
            if numel(chanExpRef{eleSec}) >= ss
                expRef = chanExpRef{eleSec}(ss);
                imBehavior(ss).CData = plotBehData{expRef}.plotData;
                imBehavior(ss).AlphaData = ~isnan(plotBehData{expRef}.plotData);
                colormap(axesBehavior(ss), plotBehData{expRef}.colorMap)
                imBehavior(ss).Parent.CLim = plotBehData{expRef}.axisLimits;
                imBehavior(ss).Parent.XLim = [0.5 numel(plotBehData{expRef}.xyValues{1})+0.5];
                imBehavior(ss).Parent.YLim = [0.5 numel(plotBehData{expRef}.xyValues{2})+0.5];
                imBehavior(ss).Parent.XTick = 1:numel(plotBehData{expRef}.xyValues{1});
                imBehavior(ss).Parent.YTick = 1:numel(plotBehData{expRef}.xyValues{2});
                imBehavior(ss).Parent.XTickLabel = plotBehData{expRef}.xyValues(1);
                imBehavior(ss).Parent.YTickLabel = plotBehData{expRef}.xyValues(2);
                axesBehavior(ss).Title.String = sprintf('%s: %d Tri, %s', ...
                    plotBehData{expRef}.subject{1}, plotBehData{expRef}.totTrials, plotBehData{expRef}.extraInf);
                
                triNum = plotBehData{expRef}.trialCount;
                [xPnts, yPnts] = meshgrid(1:size(plotBehData{expRef}.plotData,2), 1:size(plotBehData{expRef}.plotData,1));
                tIdx = ~isnan(plotBehData{expRef}.plotData);
                txtD = num2cell([xPnts(tIdx), yPnts(tIdx), round(100*plotBehData{expRef}.plotData(tIdx))/100, triNum(tIdx)],2);
                delete(textBehavior{ss})
                textBehavior{ss} = cellfun(@(x) text(x(1),x(2), {num2str(x(3)), num2str(x(4))}, 'horizontalalignment', 'center','Parent',axesBehavior(ss)), txtD);

            else
                imBehavior(ss).CData = nan;
                imBehavior(ss).AlphaData = 0;
                imBehavior(ss).Parent.CLim = [0 1];
                imBehavior(ss).Parent.XLim = [0.5 1.5];
                imBehavior(ss).Parent.YLim = [0.5 1.5];
                imBehavior(ss).Parent.XTickLabel = {'NaN'};
                imBehavior(ss).Parent.YTickLabel = {'NaN'};
                axesBehavior(ss).Title.String = 'No session';
                delete(textBehavior{ss})
                textBehavior{ss} =  text(1,1,'No trial', 'horizontalalignment', 'center','Parent',axesBehavior(ss));
            end
        end
    end
end
% close(fig);numel(chanExpRef{eleSec}) >= ss
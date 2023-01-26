function recLocation(varargin)
    %% Plot the location of the recordings (behavior) performed across the whole probe(s).
    %
    % Parameters:
    % -------------------
    % Classic PinkRigs input (optional).
    % pltClusters: bool
    %   Whether to plot clusters location along the probe.
    % clustersDays: cell of str
    %   Specifies from which dates to take the aforementioned clusters.
    % clustersExpDef: str
    %   Specifies from which expDef to take the clusters.

    %% Get parameters
    varargin = ['expDate', {inf}, varargin];
    varargin = [varargin, 'expDef', {'t'}]; % forced
    varargin = ['checkEvents', {1}, varargin];
    varargin = ['checkSpikes', {1}, varargin];
    varargin = ['pltClusters', {1}, varargin];
    varargin = ['clustersDays', {[]}, varargin];
    varargin = ['clustersExpDef', {'spontaneousActivity'}, varargin];
    params = csv.inputValidation(varargin{:});

    %% Get exp list
    
    exp2checkList = csv.queryExp(params);

    % Check it's only one subject
    if numel(unique(exp2checkList.subject))>1
        error('Please provide only one subject.')
    end

    %% Get the probes layout
    
    % Get probes layout
    probeInfo = csv.checkProbeUse(params.subject);
    probeNum = numel(probeInfo.serialNumbers{1});
    
    % Get the basic layout
    chanPosAll = [];
    chanProbeSerialNoAll = [];
    shankNum = zeros(probeNum,1);
    elecPos = cell(probeNum,1);
    probeType = cell(probeNum,1);
    probeSerialNo = nan(probeNum,1);
    for pp = 1:probeNum
        probeType{pp} = probeInfo.probeType{1}{pp};
        probeSerialNo(pp) = probeInfo.serialNumbers{1}(pp);
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

        % Get behavior
        plotBehData_tmp = plt.behaviour.boxPlots('subject',expInfo.subject, ...
            'expDate',expInfo.expDate,...
            'expDef', expInfo.expDef, ...
            'expNum', expInfo.expNum, ...
            'plotType', 'res', ...
            'noPlot', 1);

        if plotBehData_tmp{1}.totTrials > 0 
            % Save it
            plotBehData(ee) = plotBehData_tmp;

            % Get which channels were recorded from
            alignmentFile = dir(fullfile(expPath,'*alignment.mat'));
            load(fullfile(alignmentFile.folder, alignmentFile.name), 'ephys');
            ephysPaths = {ephys.ephysPath}; clear ephys

            for pp = 1:numel(ephysPaths)
                if ~strcmp(expInfo.extractSpikes{1}((pp-1)*2+1),'1')
                    fprintf('Ephys not processed %s. Skipping.\n',ephysPaths{pp})
                else
                    binFile = dir(fullfile(ephysPaths{pp},'*ap.*bin'));
                    metaData = readMetaData_spikeGLX(binFile(1).name,binFile(1).folder);

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
                    for c = 1:size(chanPosRec,1)
                        chanIdx = all(chanPosAll == chanPosRec(c,:),2) & (chanProbeSerialNoAll == probeSerialNoExp);
                        chanExpRef{chanIdx} = [chanExpRef{chanIdx} ee];
                    end
                end
            end
        end
    end
    
    %%  Build the different probe images
    
    % Total number of trials
    trialNumChan = cellfun(@(x) sum(cellfun(@(y) y.totTrials, plotBehData(x))), chanExpRef);
    imageProbes{1} = nan(ceil(max(chanPosAllPlt(:,1))/15),ceil(max(chanPosAllPlt(:,2))/15)); % 15 * 15 um map
    for c = 1:size(chanPosAllPlt,1)
        imageProbes{1}(floor(chanPosAllPlt(c,1)/15)+1,floor(chanPosAllPlt(c,2)/15)+1) = trialNumChan(c);
    end
    imageProbesLeg{1} = 'Trial count';
    
    % Total number of good sessions
    trialNumChan = cellfun(@(x) sum(cellfun(@(y) y.totTrials>200, plotBehData(x))), chanExpRef);
    imageProbes{2} = nan(ceil(max(chanPosAllPlt(:,1))/15),ceil(max(chanPosAllPlt(:,2))/15)); % 15 * 15 um map
    for c = 1:size(chanPosAllPlt,1)
        imageProbes{2}(floor(chanPosAllPlt(c,1)/15)+1,floor(chanPosAllPlt(c,2)/15)+1) = trialNumChan(c);
    end
    imageProbesLeg{2} = 'Good session count';
    
    %% Plot the basic layout
    
    fig = figure('Position', [1 31 1920 973],'Name', params.subject{1},'color','w');

    clear probeAxes behaviorAxes
    
    recNumPerChan = cellfun(@(x) size(x,2), chanExpRef);
    nCol = 4;
    nRow = max(1,ceil(max(recNumPerChan)/(nCol-1)));
    
    % Set the layout for the probes
    axesProbes = subplot(nRow,nCol,(0:nRow-1)*nCol+1); hold all
    axesProbes.Title.String = 'No channel selected';
    
    imageProbesIdx = 1;
    imProbes = imagesc(0:15:max(chanPosAllPlt(:,1)),0:15:max(chanPosAllPlt(:,2)),imageProbes{imageProbesIdx}','AlphaData',~isnan(imageProbes{imageProbesIdx}'));
    colormap(axesProbes,'parula')
    cb = colorbar;
    cb.Position = [.27 .8 .01 .1];
    cb.Title.String = imageProbesLeg{imageProbesIdx};
    xlim([-100 3000]);
    ylim([0 6000]);
    
    % Add the selected channel
    elecSelectCoord = [0 0];
    eleSecScat = scatter(elecSelectCoord(1), elecSelectCoord(2), 35, 'r', 'square','LineWidth',3);
    
    % Add a reference recording to visualize clusters density
    if params.pltClusters{1}
        clustersDays = params.clustersDays{1};
        
        % Get spontaneous recordings to plot cluster density
        paramsClu.subject = params.subject;
        paramsClu.expDef = {params.clustersExpDef};
        if ~isempty(clustersDays)
            paramsClu.expDate = {clustersDays}; % find a better way
        end
        paramsClu.checkSpikes = {'1'};
        exp2checkListClu = csv.queryExp(paramsClu);
        if isempty(clustersDays)
            fullProbeScan = {'0-0', '1-0', '2-0', '3-0', ...
                '0-2880', '1-2880', '2-2880', '3-2880'};

            % Take the last recordings, to span everything
            %%% NEED TO WRITE IT, NOT EASY TO CODE
            spannedAll = 0;
            fullProbeMatch = zeros(size(fullProbeScan,2),probeNum);
            ee = size(exp2checkListClu,1);
            while ee > 0 && ~spannedAll
                
                expInfo = exp2checkListClu(ee,:);
                expPath = expInfo.expFolder{1};
                
                alignmentFile = dir(fullfile(expPath,'*alignment.mat'));
                alignment = load(fullfile(alignmentFile.folder, alignmentFile.name), 'ephys');
                ephysPaths = {alignment.ephys.ephysPath}; 

                for pp = 1:numel(ephysPaths)
                    if strcmp(expInfo.extractSpikes{1}((pp-1)*2+1),'1')
                        binFile = dir(fullfile(alignment.ephys(pp).ephysPath,'*ap.*bin'));
                        [chanPos,~,shanks] = getRecordingSites(binFile(1).name,binFile(1).folder);
                        shankIDs = unique(shanks);
                        botRow = min(chanPos(:,2));
                        recID = [num2str(shankIDs) '-' num2str(botRow)];
                        if fullProbeMatch(strcmp(fullProbeScan,recID),pp) == 0
                            fullProbeMatch(strcmp(fullProbeScan,recID),pp) = ee;
                        end
                    end
                end

                spannedAll = all(fullProbeMatch(:)>0);
                ee = ee-1;    
            end
        end
        
        for pp = 1:probeNum
            for ee = 1:size(fullProbeScan,2)
                if ~(fullProbeMatch(ee,pp)==0)
                    expInfo = exp2checkListClu(fullProbeMatch(ee,pp),:);
                    
                    disp(expInfo.expDate);
                    clusters = csv.loadData(expInfo,dataType={sprintf('probe%d',pp-1)}, ...
                        object={'clusters'}, ...
                        attribute={{'_av_KSLabels';'_av_xpos';'depths'}});
                    KSLabels = clusters.dataSpikes{1}.(sprintf('probe%d',pp-1)).clusters.KSLabels;
                    xpos = clusters.dataSpikes{1}.(sprintf('probe%d',pp-1)).clusters.xpos;
                    depths = clusters.dataSpikes{1}.(sprintf('probe%d',pp-1)).clusters.depths;

                    goodUnits = KSLabels == 2;

                    scatter(xpos(goodUnits) + (pp-1)*sum(shankNum)*shankSep + shankSep/3, ...
                        depths(goodUnits),...
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
            textBehavior{ss} =  text(1,1,'No session','horizontalalignment', 'center','Parent',axesBehavior(ss));
            axis equal tight
        end
    end
    
    %% Interactive plot of behavior
    
    keepGoing = 1;
    while keepGoing
        was_a_key = waitforbuttonpress;
        if was_a_key && (strcmp(get(fig, 'CurrentKey'), 'leftarrow') || strcmp(get(fig, 'CurrentKey'), 'rightarrow'))
            if strcmp(get(fig, 'CurrentKey'), 'leftarrow')
                imageProbesIdx = max(imageProbesIdx-1,1);
            else
                imageProbesIdx = min(imageProbesIdx+1,numel(imageProbes));
            end
            imProbes.CData = imageProbes{imageProbesIdx}';
            cb.Title.String = imageProbesLeg{imageProbesIdx};
            cb.Limits = [0 max(imageProbes{imageProbesIdx}(:))];
            
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
                                
            % Update bot row
            axesProbes.Title.String = sprintf('Position: %d / botRow: %d',yposPlt, floor(yposPlt/15));
            
            % Update associated behavior
            for ss = 1:numel(axesBehavior)
                if numel(chanExpRef{eleSec}) >= ss
                    expRef = chanExpRef{eleSec}(ss);
                    
                    % Update Behavior plots
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
                        plotBehData{expRef}.subject, plotBehData{expRef}.totTrials, plotBehData{expRef}.extraInf);
                    
                    % Plot the performance and trial number in each condition
                    triNum = plotBehData{expRef}.trialCount;
                    [xPnts, yPnts] = meshgrid(1:size(plotBehData{expRef}.plotData,2), 1:size(plotBehData{expRef}.plotData,1));
                    tIdx = ~isnan(plotBehData{expRef}.plotData);
                    txtD = num2cell([xPnts(tIdx), yPnts(tIdx), round(100*plotBehData{expRef}.plotData(tIdx))/100, triNum(tIdx)],2);
                    delete(textBehavior{ss})
                    textBehavior{ss} = cellfun(@(x) text(x(1),x(2), {num2str(x(3)), num2str(x(4))}, 'horizontalalignment', 'center','Parent',axesBehavior(ss)), txtD);
                    
                    set(axesBehavior(ss),'Visible','on')
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
                    textBehavior{ss} =  text(1,1,'No session', 'horizontalalignment', 'center','Parent',axesBehavior(ss));
                    set(axesBehavior(ss),'Visible','off')
                end
            end
        end
    end
    % close(fig);numel(chanExpRef{eleSec}) >= ss
    
end
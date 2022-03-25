% clear all
% close all

%% Get experiments

clear params
params.mice2Check = 'AV007';
params.days2Check = 7;
% params.expDef2Check = 'spontaneousActivity';
params.expDef2Check = 'AVPassive_ckeckerboard_postactive';
params.align2Check = '*,*,*,*,*,1';
exp2checkList = csv.queryExp(params); 
pltClusters = 1;

%% Plot it

% Get probes layout
mainCSV = csv.readTable(csv.getLocation('main'));
mouseIdx = strcmpi(mainCSV.Subject,params.mice2Check);
probeNum = ~isempty(mainCSV(mouseIdx,:).P0_type) + ~isempty(mainCSV(mouseIdx,:).P1_type);

% Plot the basic layout
figure('Position', [680    32   551   964],'Name', params.mice2Check); hold all
for pp = 1:probeNum
    probeType = mainCSV(mouseIdx,:).(sprintf('P%d_type',pp-1)){1};
    switch probeType
        case '2.0 - 4shank'
            % Taken from the IMRO plotting
            shankNum(pp) = 4;
            nElec = 1280;   % per shank; pattern repeats for the four shanks
            vSep = 15;      % in um
            hSep = 32;
            shankSep = 200; % That's what's in the ephys metadata...
            elecPos = zeros(nElec, 2);
            elecPos(1:2:end,1) = 0;                %sites 0,2,4...
            elecPos(2:2:end,1) =  hSep;            %sites 1,3,5...
            % fill in y values
            viHalf = (0:(nElec/2-1))';                %row numbers
            elecPos(1:2:end,2) = viHalf * vSep;       %sites 0,2,4...
            elecPos(2:2:end,2) = elecPos(1:2:end,2);  %sites 1,3,5...
        otherwise
            error('Not yet coded')
    end
    
    for shank = 1:shankNum(pp)
        scatter((pp-1)*sum(shankNum)*shankSep + shankSep*(shank-1) + elecPos(:,1), elecPos(:,2), 30, 'k', 'square')
    end
end
xlim([-100 3000]);
ylim([0 6000]);
%axis equal tight

textLoc = [0 0];
boxLoc = [0 0 0 0];
for ee = 1:size(exp2checkList,1)
    
    % Get exp info
    expInfo = exp2checkList(ee,:);
    expPath = expInfo.expFolder{1};
    % temporarily dead
    % ephysPaths = regexp(expInfo.ephysRecordingPath,',','split');
    % ephysPaths = ephysPaths{1};
    alignmentFile = dir(fullfile(expPath,'*alignment.mat'));
    load(fullfile(alignmentFile.folder, alignmentFile.name), 'ephys');
    ephysPaths = {ephys.ephysPath};
    
    for pp = 1:numel(ephysPaths)
        binFile = dir(fullfile(ephysPaths{pp},'*ap.bin'));
        if isempty(binFile)
            fprintf('No bin file in %s. Skipping.\n',ephysPaths{pp})
        else
            metaData = readMetaData_spikeGLX(binFile.name,binFile.folder);
            
            %% Extract info from metadata
            %%% Same as in plotIMROProtocol
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
            chanPos = elecPos(elecInd+1,:);
            
            for sI = 0:3
                cc = find(shank == sI);
                chanPos(cc,1) = chanPos(cc,1) + shankSep*sI;
            end
            
            %% Plot a square around parts that were recorded from
            miXpos = min(chanPos(:,1)) + (pp-1)*sum(shankNum)*shankSep;
            maXpos = max(chanPos(:,1)) + (pp-1)*sum(shankNum)*shankSep + shankSep/3;
            miDepth = min(chanPos(:,2));
            maDepth = max(chanPos(:,2));
            
            boxLoctmp = [miXpos maXpos miDepth maDepth];
            spacer = 0;
            while any(((boxLoctmp(3) >= boxLoc(:,3) & boxLoctmp(3) <= boxLoc(:,4)) | ...
                    (boxLoctmp(4) >= boxLoc(:,3) & boxLoctmp(4) <= boxLoc(:,4)) | ...
                    (boxLoctmp(4) >= boxLoc(:,4) & boxLoctmp(3) <= boxLoc(:,3))) & ...
                    boxLoctmp(1) == boxLoc(:,1))
                spacer = spacer + 15;
                boxLoctmp = [miXpos-spacer maXpos+spacer miDepth maDepth];
            end
            boxLoc = [boxLoc; boxLoctmp];
            
            textLoctmp = [maXpos+10,maDepth];
            while any(all(textLoctmp == textLoc,2)) % Dirty way to avoid overlap
                textLoctmp = textLoctmp - [0 100];
            end
            textLoc = [textLoc; textLoctmp];
            protocolColor = getProtocolColor(expInfo.expDef{1});
            text(boxLoctmp(2),textLoctmp(2),[expInfo.expDate{1} '_' expInfo.expNum{1}],'color',protocolColor)
            
            
            rectangle('Position',[boxLoctmp(1), boxLoctmp(3), boxLoctmp(2)-boxLoctmp(1), boxLoctmp(4)-boxLoctmp(3)], ...
                'EdgeColor',protocolColor,'FaceColor',[protocolColor 0.1],'LineWidth',3)
        end
    end
end   

protocols = unique(exp2checkList.expDef);
for pro = 1:numel(protocols)
    protocolColor = getProtocolColor(protocols{pro});
    text(shankNum(1)*shankSep,6000-pro*100,regexprep(protocols{pro},'_',' '),'color',protocolColor)
end

% set (gcf, 'WindowButtonMotionFcn', @mouseBotRow);

%% Add a reference recording to visualize clusters

if pltClusters
    params.days2Check = {'2022-03-23','2022-03-22'};
    params.expDef2Check = 'spontaneousActivity';
    exp2checkList = csv.queryExp(params);
    
    qMetricFilter = 2;
    if qMetricFilter == 1
        % get good units
        bc_qualityParamValues;
        paramBC = param; clear param;
        paramBC.somatic = [0 1];
        paramBC.minAmplitude = 10;
    end
    
    for ee = 1:size(exp2checkList,1)
        preprocFile = dir(fullfile(exp2checkList(ee,:).expFolder{1},'*preprocData.mat'));
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
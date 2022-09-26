function [data, proc, recPathUni] = loadData(varargin)
    %%% This function will get the natural images data (PSTHs).
    
    %% Get parameters
    mice = csv.readTable(csv.getLocation('main'));

    % Get processing parameters
    proc.window = [-0.3 0.5 ... % around onset
        0.0 0.5]; % around offset
    proc.binSize = 0.01; % in ms
    nBins = int64((proc.window(2) - proc.window(1) + proc.window(4) - proc.window(3))/proc.binSize);
    proc.smoothSize = 5; % PSTH smoothing filter
    gw = gausswin(proc.smoothSize,3);
    proc.smWin = gw./sum(gw); 

    varargin = ['subject', {mice(contains(mice.P0_type, '2.0 - 4shank'),:).Subject}, varargin];
    varargin = ['expDate', {inf}, varargin];
    varargin = ['expDef', {{{'i'}}}, varargin]; 
    varargin = [varargin, 'checkEvents', {1}]; % forced, otherwise can't process
    varargin = [varargin, 'checkSpikes', {1}]; % forced, otherwise can't process
    varargin = ['proc', {proc}, varargin];
    params = csv.inputValidation(varargin{:});

    %% Get exp list

    exp2checkList = csv.queryExp(params);

    %% Load data

    proc = params.proc{1}; % should all be the same

    baSm = cell(1,1);
    C = cell(1,1);
    recLocAll = cell(1,1);
    recPath = cell(1,1);
    days = cell(1,1);
    nn = 1;
    for ee = 1:size(exp2checkList,1)
        fprintf('Processing experiment #%d/%d...\n',ee,size(exp2checkList,1))
        expInfo = exp2checkList(ee,:);
        subject = expInfo.subject{1};

        % Get events
        events = csv.loadData(expInfo,dataType = 'eventsFull');
        imageOnsetTimes = events.dataEvents{1}.imageOnsetTimes;
        imageOffsetTimes = events.dataEvents{1}.imageOffsetTimes;
        imageIDs = events.dataEvents{1}.imageIDs;

        % Get alignment file
        alignmentFile = dir(fullfile(expInfo.expFolder{1},'*alignment.mat'));
        alignment = load(fullfile(alignmentFile.folder,alignmentFile.name),'ephys');

        for pp = 1:numel(alignment.ephys)
            if strcmp(expInfo.extractSpikes{1}((pp-1)*2+1),'1')
                % Get recording location
                binFile = dir(fullfile(alignment.ephys(pp).ephysPath,'*ap.*bin'));
                [chanPos,~,shanks,probeSN] = getRecordingSites(binFile(1).name,binFile(1).folder);
                shankIDs = unique(shanks);
                botRow = min(chanPos(:,2));

                % Get responses
                dataExp = csv.loadData(expInfo,dataType={sprintf('probe%d',pp-1)}, ...
                        object={{'spikes'},{'clusters'}}, ...
                        attribute={{'times','clusters'}, ...
                        {'_av_IDs','_av_KSLabels','depths','_av_xpos','_av_xpos','qualityMetrics'}});
                spikes = dataExp.dataSpikes{1}.(sprintf('probe%d',pp-1)).spikes;
                clusters = dataExp.dataSpikes{1}.(sprintf('probe%d',pp-1)).clusters;
                
                if ~isfield(clusters,'qualityMetrics')
                    error('no qm')
                end

                nClusters = numel(clusters.IDs);
                nTrials = numel(imageOnsetTimes);
                baSmtmp = zeros(nTrials, nBins, nClusters);

                % get all onset-centered psths
                for c = 1:nClusters
                    temp = clusters.IDs(c);
                    st = spikes.times(spikes.clusters == temp);

                    % get psth
                    [~, ~, ~, ~, ~, baOn] = psthAndBA(st, imageOnsetTimes, proc.window(1:2), proc.binSize);
                    [~, ~, ~, ~, ~, baOff] = psthAndBA(st, imageOffsetTimes, proc.window(3:4), proc.binSize);

                    % smooth ba
                    ba = cat(2,baOn,baOff);
                    baSmtmp(:,:,c) = conv2(proc.smWin,1,ba', 'same')'./proc.binSize;
                end

                % not optimal here?
                trials = imageIDs(1:nTrials);
                trialid = unique(trials);
                baSm{nn} = nan(numel(trialid), nBins, nClusters, ceil(nTrials/numel(trialid)));
                for tt = 1:numel(trialid)
                    idxrep = trials == trialid(tt);
                    baSm{nn}(tt,:,:,1:sum(idxrep)) = permute(baSmtmp(idxrep,:,:),[2 3 1]);
                end

                C{nn} = clusters;

                % tags
                days{nn} = datenum(expInfo.expDate);
                days{nn} = days{nn}-datenum(datetime(mice(strcmp(mice.Subject,subject),:).P0_implantDate{1}(1:end),'InputFormat','yyyy-MM-dd'));
                recPath{nn} = alignment.ephys(pp).ephysPath;
                recLocAll{nn} = [subject '__' num2str(probeSN) '__' num2str(shankIDs) '__' num2str(botRow)];

                nn = nn+1;
            end
        end
    end

    % Merge recordings on the same day / same probe / same region
    % concatenate when played each repeat separately
    recPathUni = unique(recPath);
    nn = 1;
    data = struct;
    for rr = 1:numel(recPathUni)
        % Find corresponding exp
        expIdx2Keep = find(strcmp(recPath, recPathUni(rr)));

        data(nn).spikeData = cat(4,baSm{expIdx2Keep});
        % clean up to free memory usage
        baSm(expIdx2Keep) = {[]};

        ee = expIdx2Keep(1);
        data(nn).C.XPos = C{ee}.xpos;
        data(nn).C.Depth = C{ee}.depths;
        data(nn).C.CluID = C{ee}.IDs;
        data(nn).C.CluLab = C{ee}.KSLabels;
        data(nn).C.QM = C{ee}.qualityMetrics; % can't subselect them?
        data(nn).goodUnits = C{ee}.IDs(ismember([C{ee}.KSLabels],2) & squeeze(nanmean(data(nn).spikeData,[1 2 4]))'>0.1);
        data(nn).days = days{ee};
        data(nn).recLoc = recLocAll{ee};

        nn = nn + 1;
    end
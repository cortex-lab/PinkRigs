function [clusterNum, recLocAll, chanMapAll, days, expInfoAll] = clusterCount(varargin)
    %% Plot the number of clusters across days for a the specified subject(s).
    %
    % Parameters:
    % -------------------
    % Classic PinkRigs input.
    %
    % Returns: 
    % -------------------
    % clusterNum: vector
    %   Number of clusters across days.
    % recLocAll: cell of str
    %   Fingerprint of the recording locations
    % chanMapAll: cell
    %   Recording locations
    % days: vector
    %   Number of days from implantation
    % qualityMetrics: struct
    %   Quality metrics for each day
    
    %% Get parameters
    mice = csv.readTable(csv.getLocation('main'));

    varargin = ['subject', {mice(contains(mice.P0_type, '2.0 - 4shank'),:).Subject}, varargin];
    varargin = ['expDate', {inf}, varargin];
    varargin = ['expDef', {{{'s','i','n'}}}, varargin]; 
    varargin = [varargin, 'checkSpikes', {1}]; % forced, otherwise can't process
    varargin = ['pltIndiv', 0, varargin];
    varargin = ['getQM', 0, varargin];
    varargin = ['getPos', 0, varargin];
    params = csv.inputValidation(varargin{:});

    %% Get exp list

    exp2checkList = csv.queryExp(params);

    %% Get the cluster count

    bc_param = bc_qualityParamValuesForUnitMatch;

    nn = 1;
    clusterNum = [];
    recLocAll = cell(1,1);
    chanMapAll = cell(1,1);
    recPath = cell(1,1);
    days = cell(1,1);
    expInfoAll = cell(1,1);
    for ee = 1:size(exp2checkList,1)
        fprintf('Processing experiment #%d/%d.\n',ee,size(exp2checkList,1))
        expInfo = exp2checkList(ee,:);
        subject = expInfo.subject{1};

        alignmentFile = dir(fullfile(expInfo.expFolder{1},'*alignment.mat'));
        alignment = load(fullfile(alignmentFile.folder,alignmentFile.name),'ephys');

        for pp = 1:numel(alignment.ephys)
            if strcmp(expInfo.extractSpikes{1}((pp-1)*2+1),'1')
                % Get recording location
                binFile = dir(fullfile(alignment.ephys(pp).ephysPath,'*ap.*bin'));
                [chanMap{nn},~,shanks,probeSN] = getRecordingSites(binFile(1).name,binFile(1).folder);
                shankIDs = unique(shanks);
                botRow = min(chanMap{nn}(:,2));

                % Build tags etc
                days{nn} = datenum(expInfo.expDate);
                days{nn} = days{nn}-datenum(datetime(mice(strcmp(mice.Subject,subject),:).P0_implantDate{1}(1:end),'InputFormat','yyyy-MM-dd'));

                recPath{nn} = alignment.ephys(pp).ephysPath;
                recLocAll{nn} = [subject '__' num2str(probeSN) '__' num2str(shankIDs) '__' num2str(botRow)];

                attr = {'_av_KSLabels','_av_IDs', '_bc_qualityMetrics'};
                if params.getQM{1}
                    attr = cat(2,attr,{'qualityMetrics'});
                end
                if params.getPos{1}
                    attr = cat(2,attr,{'_av_xpos','depths'});
                end

                expInfoAll{nn} = csv.loadData(expInfo,dataType={sprintf('probe%d',pp-1)}, ...
                    object='clusters', ...
                    attribute=attr);
                unitQuality = bc_getQualityUnitType(bc_param,expInfoAll{nn}.dataSpikes{1}.(sprintf('probe%d',pp-1)).clusters.bc_qualityMetrics);
                goodUnits = ismember(unitQuality, [1 3]);

                % Get cluster count
                clusterNum = [clusterNum, sum(goodUnits)];

                nn = nn + 1;
            end
        end
    end

    checkIfEmpty = @(C) cellfun(@(x) isempty(x), C);
    goodIdx = ~checkIfEmpty(days);
    days = cell2mat(days(goodIdx));
    recLocAll = recLocAll(goodIdx);
    clusterNum = clusterNum(goodIdx);

    %% Plot it
    recInfo = cellfun(@(x) split(x,'__'),recLocAll,'uni',0);
    subjectsAll = cellfun(@(x) x{1}, recInfo, 'UniformOutput', false);
    probeSNAll = cellfun(@(x) x{2}, recInfo, 'UniformOutput', false);

    subjects = unique(subjectsAll);
    colAni = hsv(numel(subjects));

    fullProbeScan = {{'0__0'}, {'1__0'}, {'2__0'}, {'3__0'}, ...
        {'0__2880'}, {'1__2880'}, {'2__2880'}, {'3__2880'}};

    slopeMean = nan(numel(subjects),2);
    figure
    % figure('Position',[680   728   450   250]);
    hold all

    % Find those that match location
    % Do it by subject and probe so that it's easier to do the whole probe
    % thing...?
    recLocSlope = cell(1,1);
    b = cell(1,1);
    for ss = 1:numel(subjects)
        subjectIdx = contains(subjectsAll,subjects(ss));
        probes = unique(probeSNAll(subjectIdx));
        for pp = 1:numel(probes)
            probeIdx = contains(probeSNAll,probes(pp));
            subAndProbeIdx = find(subjectIdx & probeIdx);
            recLocGood = recLocAll(subAndProbeIdx);

            recLoc = unique(recLocGood);
            for rr = 1:numel(recLoc)
                recIdx = find(strcmp(recLocAll,recLoc{rr}));
                if numel(unique(days(recIdx)))>2

                    recLocSlope{ss,pp}{rr} = recLoc{rr};

                    % Compute the slope
                    X = [ones(numel(recIdx),1), days(recIdx)'];
                    b{ss,pp}(rr,:) = (X\log10(clusterNum(recIdx)'));

                    if params.pltIndiv{1}
                        plot(days(recIdx), clusterNum(recIdx),'-','color',[colAni(ss,:) .2])
                        scatter(days(recIdx), clusterNum(recIdx),5,colAni(ss,:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)
                        plot(days(recIdx), 10.^(X*b{ss,pp}(rr,:)'), 'color',colAni(ss,:),'LineWidth',1)
                    end
                else
                    recLocSlope{ss,pp}{rr} = '';
                    b{ss,pp}(rr,:) = [nan;nan];
                end
            end

            % 'running average' for fullProbeScan?
            fullProbeScanSpec = cellfun(@(x) [subjects{ss} '__' probes{pp} '__' x{1}], fullProbeScan, 'uni', 0);
            ee = numel(fullProbeScanSpec)+2;
            n = 1;
            clear clusterNumProbe dayFullProbe
            while ee < numel(recLocGood)
                if all(cell2mat(cellfun(@(x) ismember(x,recLocGood(ee-numel(fullProbeScanSpec)-2+1:ee)), fullProbeScanSpec, 'uni', 0)))
                    clusterNumProbe(n) = sum(clusterNum(subAndProbeIdx(ee-numel(fullProbeScanSpec)-2+1:ee)));
                    dayFullProbe(n) = days(subAndProbeIdx(ee));
                    ee = ee+numel(fullProbeScanSpec)+2;
                    n = n+1;
                else
                    ee = ee+1;
                end
            end
            if exist('clusterNumProbe','var')
                plot(dayFullProbe,clusterNumProbe,'-','color',[colAni(ss,:) .2])
                scatter(dayFullProbe,clusterNumProbe,5,colAni(ss,:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)
                X = [ones(numel(dayFullProbe),1), dayFullProbe'];
                ball = (X\log10(clusterNumProbe'));
                plot(dayFullProbe, 10.^(X*ball), 'color',colAni(ss,:),'LineWidth',2)
                text(dayFullProbe(end), 10.^(X(end,:)*ball),[subjects{ss} ' ' probes{pp}],'color',colAni(ss,:))
            end

            slopeMean(ss,pp) = nanmean(b{ss,pp}(:,2));
        end
    end

    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    ylabel({'Cluster count';''})
    xlabel('Days from implantation')
    yticks([1 10 100 1000])
    yticklabels([1 10 100 1000])
    xticks([1 5 10 25 50 100])
    xticklabels([1 5 10 25 50 100])
    ylim([1,2000])
    xlim([0,max(days)])

    % slope
    figure('Position',[680   728   200   250]);
    tmp = slopeMean(~isnan(slopeMean(:)));
    colAnitmp = [colAni(~isnan(slopeMean(:,1)),:); colAni(~isnan(slopeMean(:,2)),:)];
    [~,idx] = sort(tmp);
    scatter(1:numel(tmp),10.^(tmp(idx))-1,40,colAnitmp(idx,:),'filled');
    ylabel({'Rate of change of cluster';  ' count (%/day)'})
    xlabel('Experiment')

end
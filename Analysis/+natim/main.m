function [dball, res] = main(varargin)
    %% Plots the number of clusters across days
    %
    % Parameters
    % ------------------
    % Classic PinkRigs inputs (optional).
    %
    % Returns:
    % -------------------
    % dball: struct
    %   Data structure (output of 'natim.loadData')
    % res: struct
    %   Results structure
    %     subjectsAll: cell
    %       List of subjects
    %     Nstable: array (nDays x nDays)
    %       Number of stable clusters across days
    %     Pstable: array (nDays x nDays)
    %       Proportion of stable clusters across days
    %     dur: array (nDays x nDays)
    %       Number of days between days
    %     NstableDur: vector (n-interday intervals)
    %       Number of stable clusters for a certain IDI
    %     PstableDur: vector (n-interday intervals)
    %       Proportion of stable clusters for a certain IDI
    %     idi: vector (n-interday intervals)
    %       Interday intervals
    %     noiseCorrStructDur: cell
    %       Noise correlation structure for each day
    %     sigCorrStructDur: cell
    %       Signal correlation structure for each day
    
    %% Get parameters
    mice = csv.readTable(csv.getLocation('main'));

    varargin = ['subject', {mice(contains(mice.P0_type, '2.0 - 4shank'),:).Subject}, varargin];
    varargin = ['expDate', {inf}, varargin];
    varargin = ['expDef', {{{'i'}}}, varargin]; 
    varargin = [varargin, 'checkEvents', {1}]; % forced, otherwise can't process
    varargin = [varargin, 'checkSpikes', {1}]; % forced, otherwise can't process
    params = csv.inputValidation(varargin{:});

    %% Get exp list

    exp2checkList = csv.queryExp(params);

    %% Load data

    dball = natim.loadData(exp2checkList);

    %% Match neurons across days for all animals

    recLocUni = unique({dball.recLoc});
    recLocUni = recLocUni(cell2mat(cellfun(@(x) sum(strcmp({dball.recLoc},x))>1, recLocUni, 'uni', 0)));
    
    % Get the data across animals
    subjectsAll = cell(numel(recLocUni),1);
    Nstable = cell(numel(recLocUni),1);
    Pstable = cell(numel(recLocUni),1);
    dur = cell(numel(recLocUni),1);
    NstableDur = cell(numel(recLocUni),1);
    PstableDur = cell(numel(recLocUni),1);
    idi = cell(numel(recLocUni),1);
    sigCorrStructDur = cell(numel(recLocUni),1);
    noiseCorrStructDur = cell(numel(recLocUni),1);

    for rr = 1:numel(recLocUni)
        expIdx2Keep = find(strcmp({dball.recLoc},recLocUni{rr}));
        db = dball(expIdx2Keep);

        recInfo = split(db(1).recLoc, '__');
        subjectsAll{rr} = recInfo{1};

        % Select only the reliable neurons?
        %%% could have other metrics here
        for dd = 1:numel(db)
            s = size(db(dd).spikeData);
            resp1 = reshape(nanmean(db(dd).spikeData(:,:,:,1:2:end),[1 4]), [s(2),s(3)]);
            resp2 = reshape(nanmean(db(dd).spikeData(:,:,:,2:2:end),[1 4]), [s(2),s(3)]);
            reliability = diag(corr(resp1,resp2));

            units2Keep = squeeze(nanmean(db(dd).spikeData,[1 2 4]))>0.1 & (db(dd).C.CluLab == 2) & reliability>0.3;
            db(dd).spikeData = db(dd).spikeData(:,:,units2Keep,:);
            db(dd).C.XPos = db(dd).C.XPos(units2Keep);
            db(dd).C.Depth = db(dd).C.Depth(units2Keep);
            db(dd).C.CluID = db(dd).C.CluID(units2Keep);
            db(dd).C.CluLab = db(dd).C.CluLab(units2Keep);
        end

        % Perform CCA
        [corrWCCA, ~] = natim.computeCCA({db.spikeData});

        % Get the best matches
        [BestMatch, BestCorr] = natim.getBestMatch(corrWCCA);

        % Compute and correct for drift
        Depth = cellfun(@(x) x.Depth, {db.C}, 'uni', 0);
        DepthCorrected = natim.correctDrift(BestMatch, BestCorr, Depth);
        for i = 1:numel(db); db(i).C.DepthCorrected = DepthCorrected{i}; end

        % Get distance between clusters
        XPos = cellfun(@(x) x.XPos, {db.C}, 'uni', 0);
        BestDist = natim.getBestDist(BestMatch, XPos, DepthCorrected);

        % Match neurons across pairs of days
        [Nstable{rr}, Pstable{rr}, dur{rr}, NstableDur{rr}, PstableDur{rr}, idi{rr}, pairAcrossAll_pairsOfDays] = natim.getMatchingStability(BestMatch,BestCorr,BestDist,[db.days],[0.05 0.5 150],1);

        % Get correlation
        [sigCorrStruct,noiseCorrStruct] = natim.getCorrelationStability({db.spikeData},pairAcrossAll_pairsOfDays);
        sigCorrStructDur{rr} = nan(1,numel(idi{rr}));
        noiseCorrStructDur{rr} = nan(1,numel(idi{rr}));
        for ididx = 1:numel(idi{rr})
            sigCorrStructDur{rr}(ididx) = nanmean(sigCorrStruct(dur{rr} == idi{rr}(ididx)));
            noiseCorrStructDur{rr}(ididx) = nanmean(noiseCorrStruct(dur{rr} == idi{rr}(ididx)));
        end

        Ntot{rr} = cell2mat(cellfun(@(x) size(x,3), {db.spikeData}, 'uni', 0));

        %%% Other optional plotting functions
        % Plot the correlation
        % natim.plotCorrelationStability(sigCorrStruct,dur,[db.days]); 

        % Plot matched neurons across some days
        % [pairAcrossAll_fewDays,sigCorr,noiseCorr] = natim.plotMatchedNeuronsAcrossDays(1:5, BestMatch, BestCorr, BestDist, ...
        %     {db.spikeData}, XPos, DepthCorrected, [db.days],[0.05 0.5 150]);

    end

    subjects = unique(subjectsAll);

    %% Plot it
    colAni = lines(numel(subjects));

    % BU
    figure;
    hold all
    for rr = 1:numel(recLocUni)
        subjectIdx = strcmp(subjects,subjectsAll{rr});
        plot(idi{rr},NstableDur{rr},'-','color',[colAni(subjectIdx,:)])
        scatter(idi{rr},NstableDur{rr},10,[colAni(subjectIdx,:)],'filled')
    end

    ylabel({'Number of'; 'matched clusters'})
    xlabel('Number of days between recordings')
    set(gca, 'YScale', 'log')
    xlim([0.5 7])
    xticks([1 7])
    xticklabels([1 7])
    ylim([1 1000])
    yticks([1 10 100 1000])
    yticklabels([1 10 100 1000])

    figure;
    subplot(311); hold all
    for rr = 1:numel(recLocUni)
        subjectIdx = strcmp(subjects,subjectsAll{rr});
        plot(idi{rr},PstableDur{rr}*100,'-','color',[colAni(subjectIdx,:) .2])
        scatter(idi{rr},PstableDur{rr}*100,5,[colAni(subjectIdx,:)],'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)

        notnan = ~isnan(PstableDur{rr}); notnan(1) = 0;
        X = [ones(numel(idi{rr}(notnan)),1), log10(idi{rr}(notnan))];
        b = (X\PstableDur{rr}(notnan)'*100);
        plot(idi{rr}(2:end), X*b, 'color',colAni(subjectIdx,:),'LineWidth',2)
        text(idi{rr}(end), X(end,:)*b,[subjects(subjectIdx) '-' recLocUni{rr}],'color',colAni(subjectIdx,:))
    end
    ylim([0 100])
    xticks([1 5 10 20 30])
    xticklabels([1 5 10 20 30])
    xlim([1,30])
    ylabel({'Proportion of'; 'matched clusters'})
    xlabel('Number of days between recordings')

    subplot(312); hold all
    for rr = 1:numel(recLocUni)
        plot(idi{rr},sigCorrStructDur{rr},'-','color',[colAni(strcmp(subjects,subjectsAll{rr}),:) .2])
        scatter(idi{rr},sigCorrStructDur{rr},5,colAni(strcmp(subjects,subjectsAll{rr}),:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)

        notnan = ~isnan(sigCorrStructDur{rr}); notnan(1) = 0;
        X = [ones(numel(idi{rr}(notnan)),1), idi{rr}(notnan)];
        b = (X\sigCorrStructDur{rr}(notnan)');
        plot(idi{rr}(notnan), X*b, 'color',colAni(strcmp(subjects,subjectsAll{rr}),:),'LineWidth',2)
    end
    ylim([0 1])
    xticks([1 5 10 20 30])
    xticklabels([1 5 20 30])
    xlim([1,30])
    ylabel({'Stability of' ; 'signal correlations'})
    xlabel('Number of days between recordings')

    subplot(313); hold all
    for rr = 1:numel(recLocUni)
        plot(idi{rr},noiseCorrStructDur{rr},'-','color',[colAni(strcmp(subjects,subjectsAll{rr}),:) .2])
        scatter(idi{rr},noiseCorrStructDur{rr},5,colAni(strcmp(subjects,subjectsAll{rr}),:),'filled','MarkerEdgeAlpha',0.2,'MarkerFaceAlpha',0.2)

        notnan = ~isnan(noiseCorrStructDur{rr}); notnan(1) = 0;
        X = [ones(numel(idi{rr}(notnan)),1), idi{rr}(notnan)];
        b = (X\noiseCorrStructDur{rr}(notnan)');
        plot(idi{rr}(notnan), X*b, 'color',colAni(strcmp(subjects,subjectsAll{rr}),:),'LineWidth',2)
    end
    ylim([0 1])
    xticks([1 5 10 20 30])
    xticklabels([1 5 10 20 30])
    xlim([1,30])
    ylabel({'Stability of' ; 'noise correlations'})
    xlabel('Number of days between recordings')

    %% Save res

    res.subjectsAll = subjectsAll;
    res.Nstable = Nstable;
    res.Pstable = Pstable;
    res.dur = dur;
    res.NstableDur = NstableDur;
    res.PstableDur = PstableDur;
    res.idi = idi;
    res.noiseCorrStructDur = noiseCorrStructDur;
    res.sigCorrStructDur = sigCorrStructDur;
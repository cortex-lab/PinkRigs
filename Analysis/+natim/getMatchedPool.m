function [units2Keep, pairAcrossAll_pairsOfDays, drift] = getMatchedPool(db,plt)

    if nargin<2
        plt = 0;
    end
    
    for dd = 1:numel(db)
        s = size(db(dd).spikeData);
        % resp1 = reshape(nanmean(db(dd).spikeData(:,:,:,1:2:end),[1 4]), [s(2),s(3)]);
        % resp2 = reshape(nanmean(db(dd).spikeData(:,:,:,2:2:end),[1 4]), [s(2),s(3)]);
        resp1 = reshape(nanmean(db(dd).spikeData(:,:,:,1:2:end),[4]), [s(1)*s(2),s(3)]);
        resp2 = reshape(nanmean(db(dd).spikeData(:,:,:,2:2:end),[4]), [s(1)*s(2),s(3)]);
        reliability = diag(corr(resp1,resp2));

        units2Keep{dd} = find(squeeze(nanmean(db(dd).spikeData,[1 2 4]))>0.1 & (db(dd).C.CluLab == 2) & reliability>0.05);
        db(dd).spikeData = db(dd).spikeData(:,:,units2Keep{dd},:);
        db(dd).C.XPos = db(dd).C.XPos(units2Keep{dd});
        db(dd).C.Depth = db(dd).C.Depth(units2Keep{dd});
        db(dd).C.CluID = db(dd).C.CluID(units2Keep{dd});
        db(dd).C.CluLab = db(dd).C.CluLab(units2Keep{dd});
    end

    % Perform CCA
    [corrWCCA, ~] = natim.computeCCA({db.spikeData});

    % Get the best matches
    [BestMatch, BestCorr] = natim.getBestMatch(corrWCCA);

    % Compute and correct for drift
    Depth = cellfun(@(x) x.Depth, {db.C}, 'uni', 0);
    [DepthCorrected,drift] = natim.correctDrift(BestMatch, BestCorr, Depth);
    for i = 1:numel(db); db(i).C.DepthCorrected = DepthCorrected{i}; end

    % Get distance between clusters
    XPos = cellfun(@(x) x.XPos, {db.C}, 'uni', 0);
    BestDist = natim.getBestDist(BestMatch, XPos, DepthCorrected);

    % Match neurons across pairs of days
    [~, ~, ~, ~, ~, ~, pairAcrossAll_pairsOfDays] = natim.getMatchingStability(BestMatch,BestCorr,BestDist,[db.days],[0.05 0.5 150],plt);

end
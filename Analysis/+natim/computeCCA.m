function [corrWCCA, W] = computeCCA(spikeData,cc2keep)

    if ~exist('cc2keep','var')
        cc2keep = 1:75;
    end
    
    % perform CCA
    Xwall = [];
    idxall = [];
    clear U S V
    for k = 1:numel(spikeData)
        s = size(spikeData{k});
        X = reshape(nanmean(spikeData{k},4),[s(1)*s(2),s(3)]);
        Xz = zscore(X);
        [U{k},S{k},V{k}] = svd(X, 'econ');
        Xw = U{k}*V{k}';
        Xwall = cat(2,Xwall, Xw);
        idxall = [idxall, ones(1,s(3))*k];
    end
    [Uall,Sall,Vall] = svd(Xwall-mean(Xwall), 'econ');

    % get weight of each neuron on each day, for each component
    pUall = pinv(Uall(:,cc2keep));
    for k = 1:numel(spikeData)
        s = size(spikeData{k});
        X = reshape(nanmean(spikeData{k},4),[s(1)*s(2),s(3)]);
        W{k} = (pUall*X)'; % directly recompute a linear regression of Uall onto the data
    %     W{k} = V{k}*pinv(S{k})*Vall(idxall==k,:)*pinv(Sall); % recomputing these weights give weird stuff
    end

    %% look at correlation of all cells with all the other cells, to see if any is stable...

    corrWCCA = cell(numel(spikeData),numel(spikeData));
    for d1 = 1:numel(spikeData)
        nCd1 = size(spikeData{d1},3);
        for d2 = 1:numel(spikeData)
            nCd2 = size(spikeData{d2},3);
            corrWCCA{d1,d2} = corr(W{d1}(:,cc2keep)',W{d2}(:,cc2keep)');
        end
    end
    

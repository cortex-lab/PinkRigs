function [corrd, lags, mdif] = getCrossCorrelationMatrix(spikeData,lagbins)

    corrd = cell(numel(spikeData),1);
    lags = cell(numel(spikeData),1);
    mdif = cell(numel(spikeData),1);
    for dd = 1:numel(spikeData)
        s = size(spikeData{dd});
        tmp = permute(spikeData{dd} - nanmean(spikeData{dd},4),[2 1 4 3]);
        tmp = cat(1,nan(numel(lagbins),s(1),s(4),s(3)),tmp);
        tmp = cat(1,tmp,nan(numel(lagbins),s(1),s(4),s(3)));
        spkLinear = reshape(tmp,[size(tmp,1)*s(1)*s(4),s(3)]);

        sTot = size(spkLinear,1);
        corrdtmp  = cell(numel(lagbins),1);
        for l = 1:numel(lagbins)
            if lagbins(l)<0
                idx1 = 1:sTot+lagbins(l);
                idx2 = -lagbins(l)+1:sTot;
            else
                idx1 = lagbins(l)+1:sTot;
                idx2 = 1:sTot-lagbins(l);
            end

            Md1 = spkLinear(idx1,:);
            Md2 = spkLinear(idx2,:);
            nanidx = any(isnan(Md1) | isnan(Md2),2);
            Md1 = Md1(~nanidx,:);
            Md2 = Md2(~nanidx,:);
            corrdtmp{l} = corr(Md1,Md2);
            corrdtmp{l}(logical(eye(size(corrdtmp{l})))) = nan;
        end
        corrd{dd} = cat(3,corrdtmp{:});
        [~,lags{dd}] = max(abs(corrd{dd}),[],3);
        m = nan(size(lags{dd}));
        for ii = 1:size(lags{dd},1)
            for jj = 1:size(lags{dd},2)
                m(ii,jj) = corrd{dd}(ii,jj,lags{dd}(ii,jj));
            end
        end
        mdif{dd} = m - corrd{dd}(:,:,lagbins == 0);
    end

end
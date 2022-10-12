function [sigCorrStruct,noiseCorrStruct] = getCorrelationStability(spikeData,pairs)
    %%% Get the correlation of the population correlation structure across
    %%% days

    sigCorrStruct = nan(size(pairs,1),size(pairs,2));
    noiseCorrStruct = nan(size(pairs,1),size(pairs,2));
    for d1 = 1:size(pairs,1)
        for d2 = 1:size(pairs,2)
            if size(pairs{d1,d2},2)>3
                % get signal and noise correlation
                sigCorr_d1 = natim.getCorrelationMatrix({spikeData{d1}(:,:,pairs{d1,d2}(1,:),:)},'signal');
                noiseCorr_d1 = natim.getCorrelationMatrix({spikeData{d1}(:,:,pairs{d1,d2}(1,:),:)},'noise');
                sigCorr_d2 = natim.getCorrelationMatrix({spikeData{d2}(:,:,pairs{d1,d2}(2,:),:)},'signal');
                noiseCorr_d2 = natim.getCorrelationMatrix({spikeData{d2}(:,:,pairs{d1,d2}(2,:),:)},'noise');
                
                % correlate them across days
                strud1 = triu(sigCorr_d1{1},1); strud1 = mat2vec(strud1(triu(true(size(strud1)),1)));
                strud2 = triu(sigCorr_d2{1},1); strud2 = mat2vec(strud2(triu(true(size(strud2)),1)));
                sigCorrStruct(d1,d2) = corr(strud1,strud2);
                
                strud1 = triu(noiseCorr_d1{1},1); strud1 = mat2vec(strud1(triu(true(size(strud1)),1)));
                strud2 = triu(noiseCorr_d2{1},1); strud2 = mat2vec(strud2(triu(true(size(strud2)),1)));
                noiseCorrStruct(d1,d2) = corr(strud1,strud2);
            else
                sigCorrStruct(d1,d2) = nan;
                noiseCorrStruct(d1,d2) = nan;
            end
        end
    end
    
end
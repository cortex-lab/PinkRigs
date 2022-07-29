function corrd = getCorrelationMatrix(spikeData,type)

if ~exist('type','var')
    type = 'noise';
end

% Get correlation matrices
corrd = cell(numel(spikeData),1);
for d = 1:numel(spikeData)
    s = size(spikeData{d});
    switch type
        case 'signal'
            Md1 = reshape(nanmean(spikeData{d}(:,:,:,1:2:end),4),[s(1)*s(2),s(3)]);
            Md2 = reshape(nanmean(spikeData{d}(:,:,:,2:2:end),4),[s(1)*s(2),s(3)]);
        case 'noise'
            Md1 = spikeData{d} - nanmean(spikeData{d},4);
            Md1 = reshape(permute(Md1,[2 1 4 3]),[s(2)*s(1)*s(4),s(3)]);
            Md2 = Md1;
    end
    
    nanidx = any(isnan(Md1) | isnan(Md2),2); 
    Md1 = Md1(~nanidx,:);
    Md2 = Md2(~nanidx,:);
    corrd_tmp = corr(Md1,Md2);
    if strcmp(type,'noise')
        for ii = 1:size(corrd_tmp,1); corrd_tmp(ii,ii) = 0; end
    end
    corrd{d} = corrd_tmp;
end

end
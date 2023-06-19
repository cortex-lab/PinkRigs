function corrd = getCorrelationMatrix(spikeData,type)
    %% Gets the correlation matrix from spike data
    %
    % Parameters:
    % -------------------
    % spikeData: cell
    %   Contains the spiking data for each day
    % type: str
    %  Type of correlation to compute ('signal' or 'noise')
    %
    % Returns:
    % -------------------
    % corrd: cell
    %   Correlation matrix for each day

    if ~exist('type','var')
        type = 'noise';
    end
    
    % Get correlation matrices
    corrd = cell(numel(spikeData),1);
    for dd = 1:numel(spikeData)
        s = size(spikeData{dd});
        switch type
            case 'signal'
                Md1 = reshape(nanmean(spikeData{dd}(:,:,:,1:2:end),4),[s(1)*s(2),s(3)]);
                Md2 = reshape(nanmean(spikeData{dd}(:,:,:,2:2:end),4),[s(1)*s(2),s(3)]);
            case 'noise'
                Md1 = spikeData{dd} - nanmean(spikeData{dd},4);
                Md1 = reshape(permute(Md1,[2 1 4 3]),[s(2)*s(1)*s(4),s(3)]);
                Md2 = Md1;
        end
        
        nanidx = any(isnan(Md1) | isnan(Md2),2); 
        Md1 = Md1(~nanidx,:);
        Md2 = Md2(~nanidx,:);
        corrd_tmp = corr(Md1,Md2);
        if strcmp(type,'noise')
            corrd_tmp(logical(eye(size(corrd_tmp)))) = nan;
        end
        corrd{dd} = corrd_tmp;
    end

end
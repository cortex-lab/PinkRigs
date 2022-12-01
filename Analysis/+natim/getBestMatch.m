function [BestMatch, BestCorr] = getBestMatch(corr2check,N)
    %%% Inputs:
    %%% - corr2check contains the correlations
    %%% - N is the number of 'best matches' to output (ordered)
    %%% Ouputs:
    %%% - BestMatch is a cell (of size num. of days x num. of days) that
    %%% contains, for each cluster on day n, the N best matching clusters
    %%% on day m.
    %%% - BestCorr contains the correlation scores of the cells defined in
    %%% Bestmatch.

    %% get best correlation/distance for each neuron
    
    if ~exist('N','var')
        N = 5;
    end
    
    % check what's the correlation with all local cells on same day and next
    % day
    clear BestMatch BestCorr
    for d1 = 1:size(corr2check,1)
        for d2 = 1:size(corr2check,1)
            nCd1 = size(corr2check{d1,d2},1);
            BestMatch{d1,d2} = nan(nCd1,N);
            BestCorr{d1,d2} = nan(nCd1,N);
            for i = 1:nCd1
                [m,idxsort] = sort(corr2check{d1,d2}(i,:),'descend');
                idxsort(isnan(m)) = [];
                if ~isempty(idxsort)
                    BestMatch{d1,d2}(i,1:min(N,numel(idxsort))) = idxsort(1:min(N,numel(idxsort)));
                    BestCorr{d1,d2}(i,1:min(N,numel(idxsort))) = corr2check{d1,d2}(i,idxsort(1:min(N,numel(idxsort))));
                end
            end
        end
    end
    
end
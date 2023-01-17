function pairAcrossAll = matchNeuronsAcrossDays(dn,BestMatch,BestCorr,BestDist,p)

    if ~exist('p','var') || isempty(p)
        % hardcoded parameters
        % should be inputs
        isolim = 0.05;
        corrlim = 0.5;
        distlim = 50;
    else
        isolim = p(1);
        corrlim = p(2);
        distlim = p(3);
    end

    pairAll = cell(numel(dn)-1,1);
    % match it day by day
    for d = 1:numel(dn)-1
        d1 = dn(d);
        d2 = dn(d+1);
        Iso = BestCorr{d1,d2}(:,1)-BestCorr{d1,d2}(:,2); % quick measure of isolation (how "different" is the second best neuron's response)
        subsel = Iso>isolim & BestCorr{d1,d2}(:,1)>corrlim & BestDist{d1,d2}(:,1)<=distlim;
        pairAll{d} = [find(subsel)'; BestMatch{d1,d2}(subsel)'];
        
        % remove the clusters that are matched to the same target
        [~,unidx] = unique(pairAll{d}(2,:));
        pairAll{d} = pairAll{d}(:,unidx);
    end

    % find the ones that match across all days, with dn(1) as a ref
    pairAcrossAll = pairAll{1};
    for d = 1:numel(pairAll)-1
        [~,id1,id2] = intersect(pairAcrossAll(d+1,:),pairAll{d+1}(1,:));
        pairAcrossAll = pairAcrossAll(:,id1); % remove the ones that didn't find a match
        pairAcrossAll = cat(1,pairAcrossAll, pairAll{d+1}(2,id2)); % concatenate with the new ones
    end


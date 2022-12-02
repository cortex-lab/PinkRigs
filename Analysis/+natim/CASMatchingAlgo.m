function [goodUnits,corrProg] = CASMatchingAlgo(C,goodUnits,plt)
    %%% Match ugly and bad neurons by building a fingerprint using a 
    %%% starting 'good' population, and adding them to the 'good' 
    %%% population one by one
    %%% 'good' units: have been matched for sure
    %%% 'ugly' units: trying to match those
    %%% 'bad' units: cannot be matched

    if nargin<3
        plt = 0;
    end

    badAndUglyUnits{1} = find(~ismember(1:size(C{1},1),goodUnits{1}))';
    badAndUglyUnits{2} = find(~ismember(1:size(C{2},1),goodUnits{2}))';

    if plt
        figure;
        subplot(131)
        im1 = imagesc(C{1}(goodUnits{1},goodUnits{1}));
        axis equal tight
        cmap = colormap('RedBlue');
        colormap(cmap)
        colorbar
        caxis([-0.3 0.3])
        subplot(132)
        im2 = imagesc(C{2}(goodUnits{2},goodUnits{2}));
        axis equal tight
        cmap = colormap('RedBlue');
        colormap(cmap)
        colorbar
        caxis([-0.3 0.3])
        subplot(133)
        p = plot([0 0],[0 0]);
        ylabel('Pop corr correlation')
        xlabel('Pop size')
    end

    c = numel(goodUnits{1});
    newGoodUnit = 0;
    corrProg = nan(1,c);
    while ~isempty(newGoodUnit) && ~isempty(badAndUglyUnits{1}) && ~isempty(badAndUglyUnits{2})

        corrFingerprints = corr(C{1}(badAndUglyUnits{1},goodUnits{1})', ...
            C{2}(badAndUglyUnits{2},goodUnits{2})');

        [BestMatch, BestCorr] = natim.getBestMatch({corrFingerprints},2);

        [~,newGoodUnit] = max(BestCorr{1}(:,1));

        newGoodUnitID{1} = badAndUglyUnits{1}(newGoodUnit);
        newGoodUnitID{2} = badAndUglyUnits{2}(BestMatch{1}(newGoodUnit,1));
        goodUnits{1} = [goodUnits{1}; newGoodUnitID{1}];
        goodUnits{2} = [goodUnits{2}; newGoodUnitID{2}];
        badAndUglyUnits{1}(newGoodUnit) = [];
        badAndUglyUnits{2}(ismember(badAndUglyUnits{2},badAndUglyUnits{2}(BestMatch{1}(newGoodUnit,1)))) = []; % can have duplicate

        c = c + 1;

        Cord{1} = C{1}(goodUnits{1},goodUnits{1});
        Cord{2} = C{2}(goodUnits{2},goodUnits{2});
        x = mat2vec(Cord{1}(logical(triu(ones(size(Cord{1})),1))));
        y = mat2vec(Cord{2}(logical(triu(ones(size(Cord{2})),1))));
        idxNaN = isnan(x) | isnan(y);
        corrProg = [corrProg corr(x(~idxNaN),y(~idxNaN))];

        % update plot
        if plt && (rem(c,10) == 0)
            im1.CData = C{1}(goodUnits{1},goodUnits{1});
            im2.CData = C{2}(goodUnits{2},goodUnits{2});
            p.XData = 1:c;
            p.YData = corrProg;
            % pause
        end
    end

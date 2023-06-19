function [DepthCorrected, drift] = correctDrift(BestMatch,BestCorr,Depth,plt)
    %% Computes the drift of the recording based on matched clusters.
    %
    % Parameters:
    % -------------------
    % BestMatch: cell
    %   Contains the cell's best matches across days, as outputted by
    %   function 'natim.getBestMatch'
    % BestCorr: cell
    %   Contains the corresponding correlation for each pair, as outputted 
    %   by function 'natim.getBestMatch'
    % Depth: cell
    %   Depth of the clusters for each day
    % plt: bool
    %   Whether to plot or not
    %
    % Returns:
    % -------------------
    % DepthCorrected: cell
    %   Corrected set of depths.
    % drift: vector
    %   Overall drift

    if ~exist('plt','var')
       plt = 0;
    end

    %% compute drift

    isolim = 0.05;
    corrlim = 0.5;
    
    if plt
        figure;
    end

    clear pairAll
    dn = 1:size(BestCorr,1);
    % check drift day by day
    drift = nan(1,numel(dn)-1);
    comp = nan(1,numel(dn)-1);
    drift2 = nan(1,numel(dn)-1);
    for d = 1:numel(dn)-1
        d1 = dn(d);
        d2 = dn(d+1);
        Iso = BestCorr{d1,d2}(:,1)-BestCorr{d1,d2}(:,2); % quick measure of isolation (how "different" is the second best neuron's response)
        subsel = Iso>isolim & BestCorr{d1,d2}(:,1)>corrlim;
        pairAll{d} = [find(subsel)'; BestMatch{d1,d2}(subsel,1)'];

        % get depths
        depthsd1 = Depth{d1}(pairAll{d}(1,:))-3500;
        depthsd2 = Depth{d2}(pairAll{d}(2,:))-3500;

        % compute shift
        drift(d) = median(depthsd2-depthsd1);

        % compute overall vertical drift & compression
        %%% maybe first the interesect to be median(d2-d1) and then compute
        %%% slope?
        b = regress(depthsd2, [depthsd1 ones(size(depthsd1))]);
        drift2(d) = b(2); % this one gives some crazy values sometimes, because depends on the slope estimation -- which itself depends on outliers
        comp(d) = b(1);

        if plt
            subplot(ceil(sqrt(numel(dn)-1)),ceil(sqrt(numel(dn)-1)),d); hold all
            scatter(depthsd1,depthsd2)
            plot(depthsd1,[depthsd1 ones(size(depthsd1))]*b)
            plot(depthsd1,depthsd1,'k--')
            axis equal tight
        end

        % compute drift as a function of cluster depth??
    end

    if plt
        figure; hold all
        hist(drift)
        vline(mean(drift))
        xlabel('Drift (um/d)')
    end
    %% correct for the drift

    DepthCorrected = Depth;
    for d = 1:numel(Depth)-1
        DepthCorrected{d+1} = Depth{d+1} - sum(drift(1:d));
    end


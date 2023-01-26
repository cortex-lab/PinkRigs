function [pairAcrossAll,sigCorr,noiseCorr] = plotMatchedNeuronsAcrossDays(dn,BestMatch,BestCorr,BestDist,spikeData,XPos,Depth,days,p,neuronIdx)
    %% Plots the responses of the clusters and correlations matrix across days
    %
    % Parameters:
    % -------------------
    % dn: vector
    %   Set of days to look at
    % BestMatch: cell
    %   Contains the cell's best matches across days, as outputted by
    %   function 'natim.getBestMatch'
    % BestCorr: cell
    %   Contains the corresponding correlation for each pair, as outputted 
    %   by function 'natim.getBestMatch'
    % BestDist: cell
    %   Contains the corresponding distance for each pair, as outputted
    %   by function 'natim.getBestDist'
    % spikeData: cell
    %   Data array with binned PSTHs for all clusters for each day
    % XPos: cell
    %   X-position of the clusters for each day
    % Depth: cell
    %   Depth of the clusters for each day
    % days: cell
    %   Number of days from implant date
    % p: vector
    %   Contains the parameters used in  [isolim corrlim distlim]
    %     isolim: minimal isolation distance
    %     corrlim: minimal correlation
    %     distlim: maximal distance
    % neuronIdx: int
    %   Which example neuron to plot
    %
    % Returns:
    % -------------------
    % pairAcrossAll: cell
    %   IDs of the paired clusters across days
    % noiseCorrStructDur: cell
    %   Noise correlation structure for each day
    % sigCorrStructDur: cell
    %   Signal correlation structure for each day

    if ~exist('p','var')
        p = [];
    end
    if ~exist('neuronIdx','var')
        neuronIdx = 1;
    end

    pairAcrossAll = natim.matchNeuronsAcrossDays(dn,BestMatch,BestCorr,BestDist,p);
    nCluFinal = size(pairAcrossAll,2);

    % sorting
	clear resp
    for d = 1:numel(dn)
        resp{d} = squeeze(nanmean(spikeData{dn(d)}(:,:,pairAcrossAll(d,:),:),[2 4]));
        resp{d} = zscore(resp{d});
    end
    S = nanmean(cat(3,resp{:}),3);
    ops.iPC = 1:min(30,size(S,2));
    [neurSort, natimSort, ~] = mapTmap(S', ops);
    
    %% plot the clusters
    
    figure;
    % create axes
    clear ax
    for d = 1:numel(dn)
        ax(d) = subplot(3,numel(dn),d); hold all
        title(sprintf('Day %s',num2str(days(dn(d)))))
        axis equal tight
    end
    % plot each cell on each day
    for i = 1:nCluFinal
        col = rand(1,3);
        for d = 1:numel(dn)
            % cell on day d
            idx = pairAcrossAll(d,i);
            scatter(ax(d),XPos{dn(d)}(idx),Depth{dn(d)}(idx),70,col,'filled','MarkerFaceAlpha',.4,'MarkerEdgeAlpha',1.0)
        end
    end
    linkaxes(ax,'xy');

    % plot the signal correlation matrices
    for d = 1:numel(dn)
        subplot(3,numel(dn),numel(dn)+d)
        sigCorr(d) = natim.getCorrelationMatrix({spikeData{dn(d)}(:,:,pairAcrossAll(d,neurSort),:)},'signal');
        imagesc(sigCorr{d})
        caxis([-0.3,0.3])
        colormap('RedBlue')
        axis equal tight
    end
    
    % plot the noise correlation matrices
    for d = 1:numel(dn)
        subplot(3,numel(dn),2*numel(dn)+d)
        noiseCorr(d) = natim.getCorrelationMatrix({spikeData{dn(d)}(:,:,pairAcrossAll(d,neurSort),:)},'noise');
        imagesc(noiseCorr{d})
        caxis([-0.3,0.3])
        colormap('RedBlue')
        axis equal tight
    end

    %% plot a cluster conserved across days
    
    % raster
    figure;
    for d = 1:numel(dn)
        idx = pairAcrossAll(d,neuronIdx);
        resptmp = squeeze(nanmean(spikeData{dn(d)}(natimSort,:,idx,:),4));
        ax(d) = subplot(1,numel(dn),d);
        % imagesc(bins, 1:size(resp{d},1), resp{d});
        imagesc(resptmp);
        hold all
        vline(0)
        vline(1.0)
        cgray = flipud(colormap('gray'));
        colormap(cgray)
        caxis([0 40])
        title(sprintf('Neuron on day %s', num2str(days(dn(d)))))
        if d == 1
            ylabel('Image ID')
        end
    end
    
    % signature
    figure;
    for d = 1:numel(dn)
        idx = pairAcrossAll(d,neuronIdx);
        resptmp = squeeze(nanmean(spikeData{dn(d)}(natimSort,:,idx,:),[2 4]));
        ax(d) = subplot(1,numel(dn),d);
        imagesc(zscore(resptmp))
        % colormap('RedBlue')
        cgray = flipud(colormap('gray'));
        colormap(cgray)
        caxis([0 4])
    end
    
    % population
    figure;
    for d = 1:numel(dn)
        ax(d) = subplot(1,numel(dn),d); hold all
        imagesc(resp{d}(natimSort,neurSort))
        % colormap('RedBlue')
        cgray = flipud(colormap('gray'));
        colormap(cgray)
        caxis([-4 4])
        scatter(find(neurSort == neuronIdx),-1,30,'k','*')
        set(gca,'YDir','reverse')
        ylim([1,numel(natimSort)])
        xlim([1,numel(neurSort)])
    end
    set(get(gcf,'children'),'clipping','off')
    
    % % evolution of the trajectory
    % colDay = zeros(numel(dn),3); colDay(:,2) = linspace(0,1,numel(dn));
    % corrd_res = reshape(corrd,[size(corrd,1)*size(corrd,2),size(corrd,3)]);
    % Y = pdist(corrd_res','correlation');
    % Ys = squareform(Y);
    % D = cmdscale(Ys,2);
    % figure;
    % scatter(D(:,1),D(:,2),60,colDay,'filled')

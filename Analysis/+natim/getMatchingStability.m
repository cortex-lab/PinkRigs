function [Nstable, Pstable, dur, NstableDur, PstableDur, idi, pairAcrossAll] = getMatchingStability(BestMatch,BestCorr,BestDist,days,p,plt)

    if ~exist('p','var')
        p = [0.05 0.5 150];
    end
    
    if ~exist('plt','var')
        plt = 0;
    end
    
    dur = nan(size(BestMatch,1),size(BestMatch,1));
    Nstable = nan(size(BestMatch,1),size(BestMatch,1));
    for d1 = 1:size(BestMatch,1)
        for d2 = 1:size(BestMatch,1)
            dur(d1,d2) = abs(days(d2)-days(d1));

            pairAcrossAll{d1,d2} = natim.matchNeuronsAcrossDays([d1 d2],BestMatch,BestCorr,BestDist,p);        
            if d1~=d2
                Nstable(d1,d2) = size(pairAcrossAll{d1,d2},2);
                Pstable(d1,d2) = size(pairAcrossAll{d1,d2},2)/size(BestMatch{d1,d2},1); 
            else
                Nstable(d1,d2) = nan;
                Pstable(d1,d2) = nan;
            end
        end
    end

    idi = unique(dur(~isnan(dur)));
    clear corrStructDur NstableDur
    for ididx = 1:numel(idi)
        % corrStructDur(ididx) = nanmean(corrStruct(dur == idi(ididx)));
        NstableDur(ididx) = nanmedian(Nstable(dur == idi(ididx)));
        PstableDur(ididx) = nanmedian(Pstable(dur == idi(ididx)));
    end

    if plt
        figure('Position',[ 700   700   500   150])
        ax1 = subplot(121);
        h = imagesc(Nstable);
        set(h, 'AlphaData', ~isnan(Nstable))
        xticks(1:size(Nstable,1));
        xticklabels(days);
        yticks(1:size(Nstable,1));
        yticklabels(days);
        xlabel('day')
        ylabel('day')
        axis equal tight 
        cmap = colormap('summer'); cmap = flipud(cmap);
        colormap(ax1,cmap)
        colorbar
        c = caxis;
        caxis([0 c(2)])
        title('Number of matched clusters')

        subplot(122)
        plot(idi(2:end),NstableDur(2:end),'k','LineWidth',2)
        yl = ylim;
        ylim([0,yl(2)])
        ylabel('Number of stable clusters')
    end
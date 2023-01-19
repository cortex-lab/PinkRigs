function plotCorrelationStability(corrStruct,dur,days)
    %% Plots the number of clusters across days
    %
    % Parameters
    % ------------------
    % corrStruct: cell
    %   Correlation structure for each day (noise or signal)
    % dur: array
    %   Number of days between days
    % days: array
    %   List of dates

    idi = unique(dur(~isnan(dur)));
    corrStructDur = nan(1,numel(idi));
    for ididx = 1:numel(idi)
        corrStructDur(ididx) = nanmean(corrStruct(dur == idi(ididx)));
    end
    
    %% Plot
    figure('Position',[ 700   700   500   150])
    ax1 = subplot(121);
    h = imagesc(corrStruct);
    set(h, 'AlphaData', ~isnan(corrStruct))
    xticks(1:size(corrStruct,1));
    xticklabels(days);
    yticks(1:size(corrStruct,1));
    yticklabels(days);
    xlabel('day')
    ylabel('day')
    axis equal tight
    cmap = colormap('RedBlue');
    colormap(ax1,cmap)
    colorbar
    caxis([-1 1])
    title('Correlation of corr structure')

    subplot(122)
    plot(idi(2:end),corrStructDur(2:end),'k','LineWidth',2)
    yl = ylim;
    ylim([0 1])
    ylabel('Correlation of corr structure')
end
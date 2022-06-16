function rowsOfGrid(xData, yData, lineColors, opt)

numLines = size(yData,1);
yData = sort(yData,3);
if ~exist('lineColors', 'var') || isempty(lineColors); lineColors = repmat([0 0 0], numLines, 1); end

if ~exist('opt', 'var'); opt = struct; end

if ~isfield(opt, 'lineStyle'); opt.lineStyle = '-'; end
if ~isfield(opt, 'lineWidth'); opt.lineWidth = 3; end
if ~isfield(opt, 'Marker'); opt.Marker = '.'; end
if ~isfield(opt, 'MarkerSize'); opt.MarkerSize = 15; end
if ~isfield(opt, 'lineColors'); opt.lineColors = lineColors; end
if ~isfield(opt, 'FaceAlpha'); opt.FaceAlpha = 0.3; end
if ~isfield(opt, 'EdgeColor'); opt.EdgeColor = 'none'; end
if ~isfield(opt, 'errorType'); opt.errorType = 'patch'; end
if size(yData,3) == 1; opt.errorType = 'none'; end

for i = {'FaceAlpha', 'EdgeColor'}; patchOpt.(i{1}) = opt.(i{1}); end
for i = {'Marker', 'lineWidth', 'lineStyle', 'MarkerSize'}; lineOpt.(i{1}) = opt.(i{1}); end

for i = 1:numLines
    if strcmp(opt.errorType, 'none')
        noNan = ~isnan(yData(i,:));
        mn = yData(i,noNan);
        noNanX = xData(noNan);
    else
        noNan = ~isnan(squeeze(yData(i,:,2)));
        mn = squeeze(yData(i,noNan,2));
        up = squeeze(yData(i,noNan,3));
        lw = squeeze(yData(i,noNan,1));
        noNanX = xData(noNan);
        
        patchY = [up, lw(end:-1:1), up(1)];
        patchX = [noNanX, noNanX(end:-1:1), noNanX(1)];
        patch(patchX, patchY, opt.lineColors(i,:), patchOpt)
    end
    
    hold on
    plot(noNanX, mn, 'color', opt.lineColors(i,:), lineOpt);
end
end
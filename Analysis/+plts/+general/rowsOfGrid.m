function rowsOfGrid(xData, yData, lineColors, opt)
%% Plots the rows of a matrix with specified properties
%
% Parameters: 
% ---------------
%
% xData (required): vector
%   The values for the x-axis (one value for each column in yData)
%   
% yData (required): matrix
%   The values for the y-axis (and each row is plotted separately)
%
% lineColors (default=[0,0,0]): matrix
%   RGB values for the colour of each line (i.e. each row of yData)
%
% opt(defaults below): struct with following optional fields
%	.lineStyle(default=-)             LineStyle (MATLAB standard)
%	.lineWidth(default=3)             LineWidth (MATLAB standard)
%	.Marker(default='.')              Marker (MATLAB standard)
%	.MarkerSize(default=15)           MarkerSize (MATLAB standard)
%	.lineColors(default=[0,0,0])      lineColors (as above)
%	.FaceAlpha(default=0.3)           Transparency of error band (patch)
%	.EdgeColor(default='none')        Edge color for error band (patch)
%	.errorType(default='patch')       Errors band be a 'patch' (or 'none')




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
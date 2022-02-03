function colorMap = redBlueMap(numberOfColors)
%% Creates and M-by-3 matrix defining a colormap blue-white-red
% INPUTS(default values)
% numberOfColors(gcf colormap length)--------Number of values in colormap

% OUTPUTS
% colorMap--------------------------[colorMap, 3] colormap

if ~exist('numberOfColors', 'var'); numberOfColors = size(get(gcf,'colormap'),1); end

if (mod(numberOfColors,2) == 0)
    % From [0 0 1] to [1 1 1], then [1 1 1] to [1 0 0];
    midVal = numberOfColors*0.5;
    rVals = (0:midVal-1)'/max(midVal-1,1);
    gVals = rVals;
    rVals = [rVals; ones(midVal,1)];
    gVals = [gVals; flipud(gVals)];
    bVals = flipud(rVals);
else
    % From [0 0 1] to [1 1 1] to [1 0 0];
    midVal = floor(numberOfColors*0.5);
    rVals = (0:midVal-1)'/max(midVal,1);
    gVals = rVals;
    rVals = [rVals; ones(midVal+1,1)];
    gVals = [gVals; 1; flipud(gVals)];
    bVals = flipud(rVals);
end

colorMap = [rVals gVals bVals]; 


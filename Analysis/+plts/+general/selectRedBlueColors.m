function colorChoices = selectRedBlueColors(values, zeroTag)
%% Returns a set of red(max) to blue(min) RGB color values
%
% NOTE: when requesting colours, the colors will be scaled by the absolute
% of the largest value requested
%
%
% Parameters: 
% ---------------
% values(required): vector
%   These the values to query. +ve will be red colors, -ve will be blue.
% 
% zeroTag(default=[0.5 0.5 0.5])
%   The color of "zero" values
%
%
% Returns: 
% -----------
% colorChoices: matrix
%   An nx3 metrix of RGB values for the color-values requested


if ~exist('zeroTag', 'var'); zeroTag = 0.5*ones(1,3); end
allColors = plts.general.redBlueMap(255);
maxLength = max(abs(values));
colorChoices = zeros(length(values),3);
fractionalPosition = values./maxLength;
colorChoices(values>0,:) = allColors(128-(round(fractionalPosition(values>0)*127)),:);
colorChoices(values<0,:) = allColors(128+(round(fractionalPosition(values<0)*-127)),:);
if any(values == 0); colorChoices(values==0,:) = zeroTag; end
colorChoices = flip(colorChoices, 1);
end
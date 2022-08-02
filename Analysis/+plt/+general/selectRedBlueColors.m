function colorChoices = selectRedBlueColors(values, zeroTag)
%% A function that returns a set of red(max) to blue(min) 
% INPUTS(default values)
% values(required)-------------These are the values to get colors for. Positive values will be given red colors, negative will be given blue.
% zeroTag([0.5 0.5 0.5])-------The color of "zero" values

if ~exist('zeroTag', 'var'); zeroTag = 0.5*ones(1,3); end
allColors = plt.redBlueMap(255);
maxLength = max(abs(values));
colorChoices = zeros(length(values),3);
fractionalPosition = values./maxLength;
colorChoices(values>0,:) = allColors(128-(round(fractionalPosition(values>0)*127)),:);
colorChoices(values<0,:) = allColors(128+(round(fractionalPosition(values<0)*-127)),:);
if any(values == 0); colorChoices(values==0,:) = zeroTag; end
colorChoices = flip(colorChoices, 1);
end
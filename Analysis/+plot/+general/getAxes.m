function [axesHandle, figureHandle] = getAxes(axesOpt, idx)
%% A method for to create and/or select figure and axes for an upcoming plot based on some
% INPUTS(default values)
% axesOpt('res')--------String indicating the type of data to plot. Options are
%	.totalNumOfAxes(1)-----------------The total number of axes in the figure
%	.axesSize([400 400])---------------Size of each axes in pixels [width height]
%	.gapBetweenAxes(25)----------------Space (in pixels) between the axes
%	.btlrMargins([50 50 50 50])--------The total number of axes in the figure
%	.reposition(1)---------------------Whether to position figure at the top-right of the left-most screen
%	.numOfRows([])---------------------Specify the number of rows to use (for figures with multiple axes)
%	.numOfCols([])---------------------Specify the number of columns to use (for figures with multiple axes)

% OUTPUTS
% axesHandle-------------handle for genereated axis
% figureHandle-----------handle for the figure created/used

%% Set defaults etc.
%Set detault values
if ~exist('idx', 'var'); idx = 1; end
if ~isfield(axesOpt, 'totalNumOfAxes'); axesOpt.totalNumOfAxes = 1; end
if ~isfield(axesOpt, 'axesSize'); axesOpt.axesSize = [400 400]; end
if ~isfield(axesOpt, 'gapBetweenAxes'); axesOpt.gapBetweenAxes = 25; end
if ~isfield(axesOpt, 'btlrMargins'); axesOpt.btlrMargins = 50*ones(1,4); end
if ~isfield(axesOpt, 'reposition'); axesOpt.reposition = 1; end
if ~isfield(axesOpt, 'numOfRows'); axesOpt.numOfRows = []; end
if ~isfield(axesOpt, 'numOfCols'); axesOpt.numOfCols = []; end

%Detect the screensize and screen ratio (width/height)
screenSize = get(0,'MonitorPositions');
screenSize = screenSize(screenSize(:,1)==min(screenSize(:,1)),:);
screenRatio = round(screenSize(3)/screenSize(4));

%Assign number of rows and columns, or calculate from a combination of default columns and screen ratio.
if ~isempty(axesOpt.numOfRows); numOfRows = axesOpt.numOfRows;
elseif axesOpt.totalNumOfAxes < 4; numOfRows = 1;
else, numOfRows = find(2*screenRatio.*(1:100)>=axesOpt.totalNumOfAxes,1);
end

numOfCols = ceil(axesOpt.totalNumOfAxes/numOfRows);
figureSize = min([axesOpt.axesSize(1)*numOfCols, axesOpt.axesSize(2)*numOfRows], screenSize(3:4));

%Convert the margins to fraction of figure size (which is what matlab uses)
axesOpt.btlrMargins(1:2) = axesOpt.btlrMargins(1:2)/figureSize(2);
axesOpt.btlrMargins(3:4) = axesOpt.btlrMargins(3:4)/figureSize(1);
axesOpt.gapBetweenAxes = axesOpt.gapBetweenAxes./figureSize;

%Create axes using tight_subplot (taken from here: https://www.mathworks.com/matlabcentral/fileexchange/27991-tight_subplot-nh-nw-gap-marg_h-marg_w)
%this operates like subplot in matlab, but allows you to specify more parameters easily. Reposition if requested.
axesHandle = plot.general.tightSubplot(numOfRows, numOfCols, idx, axesOpt.gapBetweenAxes, axesOpt.btlrMargins(1:2), axesOpt.btlrMargins(3:4));
if axesOpt.reposition; set(gcf, 'position', [screenSize(1:2)+screenSize(3:4)-figureSize-[0 75], figureSize]); end
figureHandle = gcf;
end
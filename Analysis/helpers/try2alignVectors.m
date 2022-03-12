function [t1Corrected, t2Corrected] = try2alignVectors(t1, t2, diffThresh,plt)
%% A funciton that tries to align vectors of the "same time points" but there is some issue (large differences, or different numbers of points)

%INPUTS(default values)
%t1(required)----------------First set of timepoints
%t2(required)----------------Second set of timepoints
%diffThresh(calculated)------The min difference between t1 and t2 timepoints that will be considered an error

%OUTPUTS
%t1Corrected-----------------First corrected set of timepoints (after removals)
%t2Corrected-----------------Second corrected set of timepoints (after removals)
%%
%If diffThresh doesn't exist, estimate it using the mean absolute diffferences of t1 and t2 times
t1Diff = diff(t1);
t2Diff = diff(t2);
if ~exist('diffThresh', 'var')
    [~, dist] = knnsearch(t2Diff,t1Diff);
    diffThresh = max([0.25 20*mad(dist)]);
end

%Find initial offset. If shifting the first 10 points doesn't align them better, move on. Otherwise, truncate accordingly.
buff = 10;
offsetTest = cell2mat(arrayfun(@(x) [sum(abs(t1Diff((buff+1:buff*2)+x)-t2Diff(buff+1:buff*2))) x],-buff:buff, 'uni', 0)');
initialOffset = offsetTest(offsetTest(:,1)==min(offsetTest(:,1)),2);
if initialOffset < 0; t2(1:abs(initialOffset)) = [];
elseif initialOffset > 0; t1(1:initialOffset) = [];
end

%Identify the shorter timeseries of length minL and compare the first minL points of each series: "compareVect"
minL = min([length(t1) length(t2)]);
compareVect = [t1(1:minL)-(t1(1)) t2(1:minL)-t2(1)];
loopNumber = 0;

%Loop that iterates through "compareVect" and looks for mismatches in the "jumps" between the two time series. When it finds them, it deletes the
%surrounding points so the jumps are removed. If it detects more than 50, assume there are bigger problems and throw an error.
if plt
    cla;
end
while find(abs(diff(diff(compareVect,[],2)))>diffThresh,1)
    if plt
        plot(diff(diff(compareVect,[],2)), '.');
    end
    errPoint = find(abs(diff(diff(compareVect,[],2)))>diffThresh,1);
    if diff(t2(errPoint:errPoint+1)) < diff(t1(errPoint:errPoint+1))
        t2(errPoint+1) = [];
    else
        t1(errPoint+1) = [];
    end
    
<<<<<<< HEAD
    t2(max([errPoint-1,1]):min([numel(t2) errPoint+1])) = [];
    t1(max([errPoint-1,1]):min([numel(t1) errPoint+1])) = [];
=======
    t2(max([errPoint-1,1]):min([errPoint+1 numel(t2)])) = [];
    t1(max([errPoint-1,1]):min([errPoint+1 numel(t1)])) = [];
>>>>>>> 7d2ec7356d9a754476afd7afeaff40c81e6a7427
    
    minL = min([length(t1) length(t2)]);
    compareVect = [t1(1:minL)-(t1(1)) t2(1:minL)-t2(1)];
    loopNumber = loopNumber+1;
    hold on;
    if loopNumber > 100; error('Extreme alignment error'); end
end
t1Corrected = t1(1:minL);
t2Corrected = t2(1:minL);
end
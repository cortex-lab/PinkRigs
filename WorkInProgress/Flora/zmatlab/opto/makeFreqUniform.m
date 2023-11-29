function subsampledData = makeFreqUniform(inputDat, numberOfShuffles, outputData, maxFreq)
%%
if ~exist('numberOfShuffles', 'var'); numberOfShuffles = 1; end
if ~exist('maxFreq', 'var'); maxFreq = inf; end
uniValues = unique(inputDat);
uniIdx = arrayfun(@(x) find(inputDat==x), uniValues, 'uni', 0);
frqValues = cellfun(@length, uniIdx);
minThresh = min([frqValues(:); maxFreq]);
if isempty(frqValues); minThresh = []; end
frqCells = arrayfun(@(x) [ones(minThresh,1); zeros(x-minThresh,1)], frqValues, 'uni', 0);
subsampledData = repmat(inputDat, 1, numberOfShuffles);
for i = 1:length(uniValues)
    idx = uniIdx{i};
    for j = 1:numberOfShuffles
        subsampledData(idx,j) = frqCells{i}(randperm(length(frqCells{i})));
    end
end

if exist('outputData', 'var') && ~isempty(outputData)
    separatedShuffles = num2cell(subsampledData,1);
    subsampledData = cell2mat(cellfun(@(x) outputData(x>0), separatedShuffles, 'uni', 0));
end
end
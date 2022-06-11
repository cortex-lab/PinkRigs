function removeDataRow(subject, expDate, expNum)
if ~iscell(expDate); expDate = {expDate}; end
if ~iscell(expNum); expNum = {expNum}; end
csvPathMouse = csv.getLocation(subject);
if ~exist(csvPathMouse, 'file'); return; end
csvData = csv.readTable(csvPathMouse);
csvRef = cellfun(@(x,y) [x,num2str(y)], csvData.expDate, csvData.expNum, 'uni', 0);
removeRef = cellfun(@(x,y) [x,num2str(y)], expDate, expNum, 'uni', 0);
csvData(contains(csvRef,removeRef),:) = [];
csv.writeClean(csvData, csvPathMouse, 0);
end
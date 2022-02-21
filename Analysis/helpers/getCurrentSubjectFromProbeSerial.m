function matchedSubject = getCurrentSubjectFromProbeSerial(probeSerial)
%% Automatically detect the type of computer
matchedSubject = cell(length(probeSerial),1);

csvLocation = getCSVLocation('main');
csvData = readtable(csvLocation);
csvFields = fields(csvData);

serialsFromCSV = cell2mat(cellfun(@(x) csvData.(x), csvFields(contains(csvFields, 'serial')), 'uni', 0)');
implantDatesFromCSV = cellfun(@(x) csvData.(x), csvFields(contains(csvFields, 'implant')), 'uni', 0)';
implantDatesFromCSV = cat(2,implantDatesFromCSV{:});

probeDates = arrayfun(@(x) min(implantDatesFromCSV(serialsFromCSV == x)), probeSerial, 'uni', 0);
probeDates(cellfun(@isempty, probeDates)) = deal({datetime('tomorrow', 'Format', 'yyyy-MM-dd')});

csvIdx = arrayfun(@(x,y) sum(implantDatesFromCSV == x{1} & serialsFromCSV == y,2), probeDates, probeSerial, 'uni', 0);
uniqueEntryDetected = cellfun(@(x) sum(x(:)), csvIdx)==1;
if ~all(uniqueEntryDetected)
    arrayfun(@(x) fprintf('No valid entry found for serial number: %d \n', x), probeSerial(~uniqueEntryDetected));
    warning('Probe(s) not entered properly in main CSV? Will not copy associated files')  
end
   
matchedSubject(uniqueEntryDetected) = cellfun(@(x) csvData.Subject(sum(x,2)>0), csvIdx(uniqueEntryDetected));

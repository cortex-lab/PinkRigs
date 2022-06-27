function matchedSubject = getCurrentSubjectFromProbeSerial(probeSerial)
%% Determine which subject is implanted with a given probe (from serial num)
matchedSubject = cell(length(probeSerial),1);
if isnumeric(probeSerial); probeSerial = num2cell(probeSerial); end

csvData = csv.readTable(csv.getLocation('main'));
csvFields = fields(csvData);

serialsFromCSV = cellfun(@(x) csvData.(x), csvFields(contains(csvFields, 'serial')), 'uni', 0)';
serialsFromCSV = cat(2,serialsFromCSV{:});
serialsFromCSV = cell2mat(cellfun(@str2double, serialsFromCSV, 'uni', 0));

implantDatesFromCSV = cellfun(@(x) csvData.(x), csvFields(contains(csvFields, 'implantDate')), 'uni', 0)';
implantDatesFromCSV = cat(2,implantDatesFromCSV{:});
dateFormats1 = cellfun(@(x) ~isempty(regexp(x,'\d\d\d\d_\d\d_\d\d', 'once')), implantDatesFromCSV);
dateFormats2 = cellfun(@(x) ~isempty(regexp(x,'\d\d/\d\d/\d\d\d\d', 'once')), implantDatesFromCSV);
dateFormats = dateFormats1 | dateFormats2;
implantDatesFromCSV(dateFormats2) = cellfun(@(x) datestr(datenum(x,'dd/mm/yyyy'), 'yyyy_mm_dd'), implantDatesFromCSV(dateFormats2), 'UniformOutput', false);
implantDatesFromCSV(~dateFormats) = deal({[datestr(now-5000, 'yyyy_mm_dd')]});
implantDatesFromCSV = cellfun(@(x) datenum(x, 'yyyy_mm_dd'), implantDatesFromCSV);

probeDates = cellfun(@(x) max(implantDatesFromCSV(serialsFromCSV == x)), probeSerial, 'uni', 0);
probeDates(cellfun(@isempty, probeDates)) = deal({now+20});

csvIdx = cellfun(@(x,y) sum(implantDatesFromCSV == x & serialsFromCSV == y,2), probeDates, probeSerial, 'uni', 0);
uniqueEntryDetected = cellfun(@(x) sum(x(:)), csvIdx)==1;
if ~all(uniqueEntryDetected)
    cellfun(@(x) fprintf('No valid entry found for serial number: %d \n', x), probeSerial(~uniqueEntryDetected));
    warning('Probe(s) not entered properly in main CSV? Will not copy associated files')  
end
   
matchedSubject(uniqueEntryDetected) = cellfun(@(x) csvData.Subject(sum(x,2)>0), csvIdx(uniqueEntryDetected));

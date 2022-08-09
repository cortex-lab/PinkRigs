function implantDate = getImplantDateFromSubject(subject)
%% Get implantation date from subject name
if ~iscell(subject); subject = {subject}; end
csvData = csv.readTable(csv.getLocation('main'));

csvSub = csvData.Subject;
csvImp = csvData.P0_implantDate;
implantDate = cellfun(@(x) csvImp{strcmp(csvSub, x)}, subject, 'uni', 0);

validImplants = cellfun(@(x) ~isempty(regexp(x,'\d\d\d\d-\d\d-\d\d', 'once')), implantDate);
implantDate(~validImplants) = deal({'none'});
end
subject = 'AV013';
server = '\\zinu';
date = '2022-06-07';

d = dir(fullfile(server,'Subjects',subject,date,'**','ephys','**','*.ap.*bin'));
clear recList
for idx = 1:numel(d)
    recList{idx} = fullfile(d(idx).folder,d(idx).name);
end

% params.mice2Check = 'AV009';
% exp2checkList = csv.queryExp(params);
% idx = 1;
% for ee = 1:size(exp2checkList,1)
%     alignmentFile = dir(fullfile(exp2checkList(ee,:).expFolder{1}, '*alignment.mat'));
%     alignment = load(fullfile(alignmentFile.folder, alignmentFile.name),'ephys');
%     for p = 1:numel(alignment.ephys)
%         recList{idx} = alignment.ephys(p).ephysPath;
%         idx = idx +1;
%     end
% end
% recList = unique(recList);

%%
params.recomputeKilo = 0;
params.recomputeQMetrics = 0;
kilo.main(params,recList)

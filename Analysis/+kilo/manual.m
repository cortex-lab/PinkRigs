subject = 'AV009';
server = '\\zinu';
date = '2022-02-23';

d = dir(fullfile(server,'Subjects',subject,date,'ephys','**','*.ap.bin'));
clear recList
for idx = 1:numel(d)
    recList{idx} = fullfile(d(idx).folder,d(idx).name);
end

%%
kilo.main([],recList)

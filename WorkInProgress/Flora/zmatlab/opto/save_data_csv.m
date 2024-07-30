
%this function allows me to process and 
clc; clear all;

set_name = 'uni_all_nogo';
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1,'whichSet', set_name);

%
basefolder = ['D:\LogRegression\opto', '\', set_name]; 

if ~exist(basefolder, 'dir')
    mkdir(basefolder);
end
%%
for i=1:numel(extracted.subject)
    hemisphere = extracted.hemisphere{i, 1};
    if hemisphere==1
        hemisphereID = 'right'; 
    elseif hemisphere==-1
        hemisphereID = 'left'; 
    end 

    power = extracted.power{i, 1}; 
    
    namestring = sprintf('%s_%s_%.0dmW.csv',extracted.subject{i, 1},hemisphereID,power);
    table = struct2table(extracted.data{i, 1});
    csv.writeTable(table,[basefolder,'\',namestring])
end
%%
% extract and save the data for the ddm fitting 

clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',0,'sepHemispheres',1,'sepPowers',1); 


%%

data_folder = 'C:\Users\Flora\Documents\Processed data\ddm\Opto\Data\bilateral'; 

for s=1:numel(extracted.subject)
    if extracted.hemisphere{s,1}==1
        hem = 'Right'; 
    elseif extracted.hemisphere{s,1}==-1
        hem = 'Left'; 
    else
        hem = 'Bi';
        extracted.power{s,1}=extracted.power{s,1}/2;
    end 
    csv_name = sprintf('%s_%.0dmW_%s.csv',extracted.subject{s,1},extracted.power{s,1},hem); 
    csv.writeTable(struct2table(extracted.data{s, 1}), [data_folder '\' csv_name])
end
%%
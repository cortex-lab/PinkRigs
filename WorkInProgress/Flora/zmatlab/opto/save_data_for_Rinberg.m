%%
clc; clear all; close all;
 addpath(genpath('C:\Users\Flora\Documents\Github\PinkRigs'));
 addpath(genpath('C:\Users\Flora\Documents\Github\2023_CoenSit'));


%% this loads everything!!
extracted = loadOptoData('balanceTrials',0,'sepMice',0,'reExtract',1,'sepHemispheres',0,'sepPowers',0,'sepDiffPowers',0,'whichSet','bi_rinberg'); 
%%
% load and format the cortex data 

[sites,cortexDat] = loadCortexData('uni',0);

% actually this is if I fit the bilateral data, which I won't in the first
% round
% [sites_bilateral,cortexDat_bilateral] = loadCortexData('bi',0);
% 
% % now we need to unify the ID-ing and then concatenate all 
% 
% cortexDat = cell(1,3);
% for site=1:numel(sites_bilateral)
%     d1 = cortexDat_unilateral{1,site}; 
%     d2 = cortexDat_bilateral{1,site};
%     
%     fields = fieldnames(d1); 
%     dat = struct();
%     for i=1:numel(fields)
%         field = fields{i}; 
%         dat.(field) = [d1.(field);d2.(field)];
%     end 
%     [unique_subjects,~,uniqueID] = unique(dat.subjectName); 
%     
%     dat.subjectID_ =  uniqueID;
%     
%     cortexDat{1,site} = dat; 
% end 
% sites = sites_bilateral;  

%%
% load the opto data for the SC (
%
sites{4} = 'SC'; 
allDat = [cortexDat,extracted.data]; 



%%

% Desired new order for the sites
new_order = {'SC','Frontal', 'Vis','Lateral'};

% Initialize new data cell array
new_data = cell(1, 3);
% Loop through the new order and rearrange the data accordingly
for i = 1:length(new_order)
    % Find the index in the original 'sites' cell that matches the current site in 'new_order'
    idx = find(strcmp(sites, new_order{i}));
    
    % Assign the corresponding data to the new data cell
    new_data{i} = allDat{idx};
end

% Update the original sites and data with the new order
sites = new_order';
allDat = new_data;




 %%
 % save out the data!


basefolder = ['D:\LogRegression\opto', '\', 'Rinberg']; 

if ~exist(basefolder, 'dir')
    mkdir(basefolder);
end
%
for i=1:numel(allDat)
    namestring = sprintf('%s.csv',sites{i});
    table = struct2table(allDat{i});
    csv.writeTable(table,[basefolder,'\',namestring])
end


clc; clear;
% TO EDIT to access file %%%%
% to map the SC: 
% 1. please select the most posteror, medial shank. This might vary dependig on how oriented the insertion of the probe and will be different on imec0 and imec1. 
% The depth of SC varies but you can expect it 1-2 mm from surface so select bank accordindly. 
% 2. Run SparseNoise protocol for 10 mins
% 
RID.mname='AV014'; % mouse name 
RID.date='2022-06-06'; % date
RID.SparseNoise_expnum=1; % expnum related to sparseNoise experiment
RID.root=sprintf('\\\\zinu.cortexlab.net\\Subjects\\%s\\%s',RID.mname,RID.date); % where to find block and timeline for sparsenoise 
RID.ephys_name=sprintf('%s_%s_SparseNoise_Part1_g0',RID.mname,RID.date); % ephys recording name
%RID.ephys_name=sprintf('FT008_RFmap_shank0_g2');
RID.probename='imec1';
RID.ephys_folder=['D:\\ephysData\\' RID.ephys_name]; % local save folder
%where you can isntantly access ephys data
%rig 
% RID.ephys_folder=[RID.root '\ephys\' RID.ephys_name]; 

RID.Githubfolder = 'C:\Users\Experiment\Documents\Github';

RF_online(RID); 
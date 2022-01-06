


% the function will take the csvs and kilosort the recent data 
% one can also set kilosort between certain dates 
% or kilosort only certain mice % e.g. only on FT mice or FT032
% fn run_kilosort(varargin)

clc; clear;
csvRoot='\\zserver.cortexlab.net\Code\AVrig\';

% check active mice
csvLocation = [csvRoot 'aMasterMouseList.csv'];
csvData = readtable(csvLocation);
mice2Check = csvData.Subject(csvData.IsActive>0);

% check recordings that have ephys
ct=1;
for subject = mice2Check'
    currSub = subject{1};
    csvPathMouse = [csvRoot currSub '.csv'];
    csvDataMouse = readtable(csvPathMouse);   
    recs2Sort(ct,1) = subject; 
    recs2Sort(ct,2) = unique(csvDataMouse.expDate(csvDataMouse.ephys>0));
    ct=ct+1;
end

% do it in python

% check whether these have been sorted

% dump into a csv the mouse name and dates that need to be sorted... ? 


%recs2Sort =
%params.dates
%params.Subjects

% if ~isempty(varargin)
%     params = varargin{1};
% end 

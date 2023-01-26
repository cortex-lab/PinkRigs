%% Define base path to generate the IMRO
clc;clear;
basePath = '\\zserver.cortexlab.net\code\Rigging\ExpDefinitions\PinkRigs\IMROFiles\AV024';

SCsurface_probe0=0; 
SCsurface_probe1=0; 

Spontaneous_botrow = 96; 

include_probe1 = 1; 

%% DAY 1
d = 1;
days{d} = '2022-10-18'; 
% PROTOCOL 1
p = 1;
imroprop{d}{p}.protocol = 'postactive_low_threshold_part1';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'hs4';
imroprop{d}{p}.probe(1).botRow = 118;
imroprop{d}{p}.probe(1).shankChoice = [0 1];
imroprop{d}{p}.probe(1).refElec = 1;

if include_probe1==1
% probe 1
imroprop{d}{p}.probe(2).patternTag = 'hs4';
imroprop{d}{p}.probe(2).botRow = 144;
imroprop{d}{p}.probe(2).shankChoice = [2 3];
imroprop{d}{p}.probe(2).refElec = 1;
end

 p = p+1;
imroprop{d}{p}.protocol = 'postactive_low_threshold_part2';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'hs4';
imroprop{d}{p}.probe(1).botRow = 70;
imroprop{d}{p}.probe(1).shankChoice = [0 1];
imroprop{d}{p}.probe(1).refElec = 1;

if include_probe1==1
% probe 1
imroprop{d}{p}.probe(2).patternTag = 'hs4';
imroprop{d}{p}.probe(2).botRow = 96;
imroprop{d}{p}.probe(2).shankChoice = [2 3];
imroprop{d}{p}.probe(2).refElec = 1;
end
% imroprop{d}{p}.protocol = 'postactive_part2';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'hs2';
% imroprop{d}{p}.probe(1).botRow = SCsurface_probe0;
% imroprop{d}{p}.probe(1).shankChoice = [2 3];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'hs2';
% imroprop{d}{p}.probe(2).botRow = SCsurface_probe0;
% imroprop{d}{p}.probe(2).shankChoice = [0 1];
% imroprop{d}{p}.probe(2).refElec = 1;
% end

% p = p+1;
% imroprop{d}{p}.protocol = 'PassiveExtended_part3';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'hs4';
% imroprop{d}{p}.probe(1).botRow = SCsurface_probe0-96;
% imroprop{d}{p}.probe(1).shankChoice = [2 3];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'hs4';
% imroprop{d}{p}.probe(2).botRow = SCsurface_probe0-96;
% imroprop{d}{p}.probe(2).shankChoice = [1 2];
% imroprop{d}{p}.probe(2).refElec = 1;
% end
% 
% p = p+1;
% imroprop{d}{p}.protocol = 'PassiveExtended_part4';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'hs4';
% imroprop{d}{p}.probe(1).botRow = SCsurface_probe0-144;
% imroprop{d}{p}.probe(1).shankChoice = [2 3];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'hs4';
% imroprop{d}{p}.probe(2).botRow = SCsurface_probe0-144;
% imroprop{d}{p}.probe(2).shankChoice = [1 2];
% imroprop{d}{p}.probe(2).refElec = 1;
% end
% 
% p = p+1;
% imroprop{d}{p}.protocol = 'PassiveExtended_part5';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'hs4';
% imroprop{d}{p}.probe(1).botRow = SCsurface_probe0-192;
% imroprop{d}{p}.probe(1).shankChoice = [2 3];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'hs4';
% imroprop{d}{p}.probe(2).botRow = SCsurface_probe0-192;
% imroprop{d}{p}.probe(2).shankChoice = [1 2];
% imroprop{d}{p}.probe(2).refElec = 1;
% end

% PROTOCOL 2
% PROTOCOL 4
% p = p+1;
% imroprop{d}{p}.protocol = 'NatImagesSparseNoise';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'hs4';
% imroprop{d}{p}.probe(1).botRow = SCsurface_probe0;
% imroprop{d}{p}.probe(1).shankChoice = [2 3];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'hs4';
% imroprop{d}{p}.probe(2).botRow = SCsurface_probe1;
% imroprop{d}{p}.probe(2).shankChoice = [1 2];
% imroprop{d}{p}.probe(2).refElec = 1;
% end 

%PROTOCOL 3
% 
% Spontaneous_botrow = 192; 
% p = p+1;
% imroprop{d}{p}.protocol = 'SparseNoise_Part1';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'sin';
% imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(1).shankChoice = [0];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'sin';
% imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(2).shankChoice = [0];
% imroprop{d}{p}.probe(2).refElec = 1;
% end
% p = p+1;
% imroprop{d}{p}.protocol = 'SparseNoise_Part2';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'sin';
% imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(1).shankChoice = [1];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'sin';
% imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(2).shankChoice = [1];
% imroprop{d}{p}.probe(2).refElec = 1;
% end 
% p = p+1;
% imroprop{d}{p}.protocol = 'SparseNoise_Part3';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'sin';
% imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(1).shankChoice = [2];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'sin';
% imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(2).shankChoice = [2];
% imroprop{d}{p}.probe(2).refElec = 1;
% end 
% 
% p = p+1;
% imroprop{d}{p}.protocol = 'SparseNoise_Part4';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'sin';
% imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(1).shankChoice = [3];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'sin';
% imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(2).shankChoice = [3];
% imroprop{d}{p}.probe(2).refElec = 1;
% end 
% 
% Spontaneous_botrow = 0; 
% p = p+1;
% imroprop{d}{p}.protocol = 'SparseNoise_Part5';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'sin';
% imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(1).shankChoice = [0];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'sin';
% imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(2).shankChoice = [0];
% imroprop{d}{p}.probe(2).refElec = 1;
% end
% p = p+1;
% imroprop{d}{p}.protocol = 'SparseNoise_Part6';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'sin';
% imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(1).shankChoice = [1];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'sin';
% imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(2).shankChoice = [1];
% imroprop{d}{p}.probe(2).refElec = 1;
% end 
% p = p+1;
% imroprop{d}{p}.protocol = 'SparseNoise_Part7';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'sin';
% imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(1).shankChoice = [2];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'sin';
% imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(2).shankChoice = [2];
% imroprop{d}{p}.probe(2).refElec = 1;
% end 
% 
% p = p+1;
% imroprop{d}{p}.protocol = 'SparseNoise_Part8';
% % probe 0
% imroprop{d}{p}.probe(1).patternTag = 'sin';
% imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(1).shankChoice = [3];
% imroprop{d}{p}.probe(1).refElec = 1;
% 
% if include_probe1==1
% % probe 1
% imroprop{d}{p}.probe(2).patternTag = 'sin';
% imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
% imroprop{d}{p}.probe(2).shankChoice = [3];
% imroprop{d}{p}.probe(2).refElec = 1;
% end 


%% Generate the protocol and plot

% Will generate the protocol
imro.generateIMROProtocol(basePath,imroprop,days)

% Will read and plot it
plts.plotIMROProtocol(basePath,1,days)

% Copy the file that was used to generate this protocol
FileNameAndLocation = mfilename('fullpath');
[path,file] = fileparts(FileNameAndLocation);
fileName = [regexprep(file,'example',protocolName) '_' regexprep(strjoin(days),' ','_') '.m'];
copyfile([FileNameAndLocation '.m'], fullfile(basePath,fileName));
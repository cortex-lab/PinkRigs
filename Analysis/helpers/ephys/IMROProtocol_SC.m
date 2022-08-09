%% Define base path to generate the IMRO
clc;clear;
basePath = '\\zserver.cortexlab.net\code\Rigging\ExpDefinitions\PinkRigs\IMROFiles\AV014';

SCsurface_probe0=198; 
SCsurface_probe1=236; 

Spontaneous_botrow = 0; 

include_probe1 = 1; 

%% DAY 1
d = 1;
days{d} = '2022-06-27'; 
% PROTOCOL 1
p = 1;
imroprop{d}{p}.protocol = 'ActivePassive';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'hs4';
imroprop{d}{p}.probe(1).botRow = SCsurface_probe0-198;
imroprop{d}{p}.probe(1).shankChoice = [2 3];
imroprop{d}{p}.probe(1).refElec = 1;

if include_probe1==1
% probe 1
imroprop{d}{p}.probe(2).patternTag = 'hs4';
imroprop{d}{p}.probe(2).botRow = SCsurface_probe1-198;
imroprop{d}{p}.probe(2).shankChoice = [1 2];
imroprop{d}{p}.probe(2).refElec = 1;
end

% PROTOCOL 2
% PROTOCOL 4
p = p+1;
imroprop{d}{p}.protocol = 'NatImagesSparseNoise';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'hs4';
imroprop{d}{p}.probe(1).botRow = SCsurface_probe0;
imroprop{d}{p}.probe(1).shankChoice = [2 3];
imroprop{d}{p}.probe(1).refElec = 1;

if include_probe1==1
% probe 1
imroprop{d}{p}.probe(2).patternTag = 'hs4';
imroprop{d}{p}.probe(2).botRow = SCsurface_probe1;
imroprop{d}{p}.probe(2).shankChoice = [1 2];
imroprop{d}{p}.probe(2).refElec = 1;
end 

% PROTOCOL 3
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part1';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
imroprop{d}{p}.probe(1).shankChoice = [0];
imroprop{d}{p}.probe(1).refElec = 1;

if include_probe1==1
% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
imroprop{d}{p}.probe(2).shankChoice = [0];
imroprop{d}{p}.probe(2).refElec = 1;
end
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part2';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
imroprop{d}{p}.probe(1).shankChoice = [1];
imroprop{d}{p}.probe(1).refElec = 1;

if include_probe1==1
% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
imroprop{d}{p}.probe(2).shankChoice = [1];
imroprop{d}{p}.probe(2).refElec = 1;
end 
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part3';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
imroprop{d}{p}.probe(1).shankChoice = [2];
imroprop{d}{p}.probe(1).refElec = 1;

if include_probe1==1
% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
imroprop{d}{p}.probe(2).shankChoice = [2];
imroprop{d}{p}.probe(2).refElec = 1;
end 

p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part4';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = Spontaneous_botrow;
imroprop{d}{p}.probe(1).shankChoice = [3];
imroprop{d}{p}.probe(1).refElec = 1;

if include_probe1==1
% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = Spontaneous_botrow;
imroprop{d}{p}.probe(2).shankChoice = [3];
imroprop{d}{p}.probe(2).refElec = 1;
end 



%% Generate the protocol and plot

% Will generate the protocol
generateIMROProtocol(basePath,imroprop,days)

% Will read and plot it
plotIMROProtocol(basePath,1,days)

% Copy the file that was used to generate this protocol
FileNameAndLocation = mfilename('fullpath');
[path,file] = fileparts(FileNameAndLocation);
fileName = [regexprep(file,'example',protocolName) '_' regexprep(strjoin(days),' ','_') '.m'];
copyfile([FileNameAndLocation '.m'], fullfile(basePath,fileName));
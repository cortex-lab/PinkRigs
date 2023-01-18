%% Define base path to generate the IMRO

basePath = '\\zserver.cortexlab.net\code\Rigging\ExpDefinitions\PinkRigs\IMROFiles\anatmap';
clear imroprop

%% DAY 1
d = 1;
% PROTOCOL 1
p = 1;
imroprop{d}{p}.protocol = 'anatmap_Part1';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 0;
imroprop{d}{p}.probe(1).shankChoice = 0;
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 0;
imroprop{d}{p}.probe(2).shankChoice = 0;
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 2
p = p+1;
imroprop{d}{p}.protocol = 'anatmap_Part2';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 0;
imroprop{d}{p}.probe(1).shankChoice = 1;
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 0;
imroprop{d}{p}.probe(2).shankChoice = 1;
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 3
p = p+1;
imroprop{d}{p}.protocol = 'anatmap_Part3';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 0;
imroprop{d}{p}.probe(1).shankChoice = 2;
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 0;
imroprop{d}{p}.probe(2).shankChoice = 2;
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 4
p = p+1;
imroprop{d}{p}.protocol = 'anatmap_Part4';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 0;
imroprop{d}{p}.probe(1).shankChoice = 3;
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 0;
imroprop{d}{p}.probe(2).shankChoice = 3;
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 5
p = p+1;
imroprop{d}{p}.protocol = 'anatmap_Part5';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = 0;
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 192;
imroprop{d}{p}.probe(2).shankChoice = 0;
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 6
p = p+1;
imroprop{d}{p}.protocol = 'anatmap_Part6';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = 1;
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 192;
imroprop{d}{p}.probe(2).shankChoice = 1;
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 7
p = p+1;
imroprop{d}{p}.protocol = 'anatmap_Part7';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = 2;
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 192;
imroprop{d}{p}.probe(2).shankChoice = 2;
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 8
p = p+1;
imroprop{d}{p}.protocol = 'anatmap_Part8';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = 3;
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 192;
imroprop{d}{p}.probe(2).shankChoice = 3;
imroprop{d}{p}.probe(2).refElec = 1;

%% Generate the protocol and plot

% Will generate the protocol
generateIMROProtocol(basePath,imroprop)

% Will read and plot it
plotIMROProtocol(basePath,1)

% Copy the file that was used to generate this protocol
FileNameAndLocation = mfilename('fullpath');
[path,file] = fileparts(FileNameAndLocation);
[~,protocolName] = fileparts(basePath);
copyfile([FileNameAndLocation '.m'], fullfile(basePath,[regexprep(file,'example',protocolName) '.m']));
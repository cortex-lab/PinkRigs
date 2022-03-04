%% Define base path to generate the IMRO

basePath = '\\zserver.cortexlab.net\code\Rigging\ExpDefinitions\PinkRigs\IMROFiles\AV010';

%% DAY 1
d = 1;
% PROTOCOL 1
p = 1;
imroprop{d}{p}.protocol = 'PassiveActive';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'hs2';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = [2 3];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'hs2';
imroprop{d}{p}.probe(2).botRow = 144;
imroprop{d}{p}.probe(2).shankChoice = [1 2];
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 2
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part1';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 0;
imroprop{d}{p}.probe(1).shankChoice = [2];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 192;
imroprop{d}{p}.probe(2).shankChoice = [0];
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 3
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part2';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = [2];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 192;
imroprop{d}{p}.probe(2).shankChoice = [1];
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 4
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part3';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 0;
imroprop{d}{p}.probe(1).shankChoice = [3];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 192;
imroprop{d}{p}.probe(2).shankChoice = [2];
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 5
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part4';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = [3];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 192;
imroprop{d}{p}.probe(2).shankChoice = [3];
imroprop{d}{p}.probe(2).refElec = 1;

%% DAY 2
d = d+1;
% PROTOCOL 1
p = 1;
imroprop{d}{p}.protocol = 'PassiveActive';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'hs2';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = [0 1];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'hs2';
imroprop{d}{p}.probe(2).botRow = 48;
imroprop{d}{p}.probe(2).shankChoice = [1 2];
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 2
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part1';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 0;
imroprop{d}{p}.probe(1).shankChoice = [0];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 0;
imroprop{d}{p}.probe(2).shankChoice = [0];
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 3
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part2';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = [0];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 0;
imroprop{d}{p}.probe(2).shankChoice = [1];
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 4
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part3';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 0;
imroprop{d}{p}.probe(1).shankChoice = [1];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 0;
imroprop{d}{p}.probe(2).shankChoice = [2];
imroprop{d}{p}.probe(2).refElec = 1;

% PROTOCOL 5
p = p+1;
imroprop{d}{p}.protocol = 'Spontaneous_Part4';
% probe 0
imroprop{d}{p}.probe(1).patternTag = 'sin';
imroprop{d}{p}.probe(1).botRow = 192;
imroprop{d}{p}.probe(1).shankChoice = [1];
imroprop{d}{p}.probe(1).refElec = 1;

% probe 1
imroprop{d}{p}.probe(2).patternTag = 'sin';
imroprop{d}{p}.probe(2).botRow = 0;
imroprop{d}{p}.probe(2).shankChoice = [3];
imroprop{d}{p}.probe(2).refElec = 1;


%% Generate the protocol and plot

% Will generate the protocol
generateIMROProtocol(basePath,imroprop)

% Will read and plot it
plotIMROProtocol(basePath,1)
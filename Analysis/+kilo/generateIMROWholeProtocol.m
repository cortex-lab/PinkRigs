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
plotIMROProtocol(basePath)

%% To generate IMRO files and save them with the right architecture.

function generateIMROProtocol(basePath,imroprop)
    if exist(basePath,'dir')
        % just save it somewhere else
        today = datestr(now);
        movefile(basePath, regexprep([basePath '_' today]),' |:','_')
    end
    
    mkdir(basePath)
    
    % Loop through days and protocols
    for d = 1:numel(imroprop)
        mkdir(fullfile(basePath,sprintf(sprintf('Day%d',d))))
        for p = 1:numel(imroprop{d})
            % Get protocol
            protocol = imroprop{d}{p}.protocol;
            savePath = fullfile(basePath,sprintf('Day%d',d),protocol);
            mkdir(savePath)
            for probeNum = 1:numel(imroprop{d}{p}.probe)
                probeProp = imroprop{d}{p}.probe(probeNum);
                prefix = sprintf('Day%d_%s_probe%d_',d,protocol,probeNum-1);
                fileName = kilo.generateIMRO_P24(probeProp.patternTag, probeProp.botRow, probeProp.shankChoice, probeProp.refElec, savePath);
                [path,file,ext] = fileparts(fileName);
                movefile(fileName,fullfile(path,[prefix file ext]))
            end
        end
    end
end

%% to plot the IMRO protocol

function plotIMROProtocol(basePath)
    probeColor = [0.4 0.6 0.2; ...
                    0.9 0.3 0.0];
    
    days = dir(basePath);
    days(~[days.isdir]) = [];
    days(ismember({days.name},{'.','..'})) = [];
    f = figure('Position',[1 30 1920 970]);
    for d = 1:numel(days)
        protocols = dir(fullfile(days(d).folder,days(d).name)); protocols(ismember({protocols.name},{'.','..'})) = [];
        for p = 1:numel(protocols)
            subplot(numel(days),numel(protocols),(d-1)*numel(protocols)+p); hold all
            IMROfiles = dir(fullfile(protocols(p).folder,protocols(p).name,'**','*.imro'));
            for probeNum = 1:numel(IMROfiles)
                %% read data
                IMROfileName = fullfile(IMROfiles(probeNum).folder,IMROfiles(probeNum).name);
                [fid,msg] = fopen(IMROfileName,'rt');
                assert(fid>=3,msg) % ensure the file opened correctly.
                out = fgetl(fid); % read and ignore the very first line.
                fclose(fid);
                out = regexp(out,'\(|\)(|\)','split'); 
                out(1:2) = []; % empty + extra channel or something?
                out(end) = []; % empty
                
                % Get channel properties
                chans = nan(1,numel(out));
                shank = nan(1,numel(out));
                bank = nan(1,numel(out));
                elecInd = nan(1,numel(out));
                for c = 1:numel(out)
                    chanProp = regexp(out{c},' ','split');
                    chans(c) = str2double(chanProp{1});
                    shank(c) = str2double(chanProp{2});
                    bank(c) = str2double(chanProp{3});
                    % 4 is refElec
                    elecInd(c) = str2double(chanProp{5});
                end
                
                %% plot data -- taken from IMRO generation script 
                % NP 2.0 MS (4 shank), probe type 24 electrode positions
                nElec = 1280;   %per shank; pattern repeats for the four shanks
                vSep = 15;      % in um
                hSep = 32;
                
                elecPos = zeros(nElec, 2);
                
                elecPos(1:2:end,1) = 0;                %sites 0,2,4...
                elecPos(2:2:end,1) =  hSep;            %sites 1,3,5...
                
                % fill in y values
                viHalf = (0:(nElec/2-1))';                %row numbers
                elecPos(1:2:end,2) = viHalf * vSep;       %sites 0,2,4...
                elecPos(2:2:end,2) = elecPos(1:2:end,2);  %sites 1,3,5...
                
                chanPos = elecPos(elecInd+1,:);
                
                % make a plot of all the electrode positions
                shankSep = 250;
                for sI = 0:3
                    cc = find(shank == sI);
                    scatter((probeNum-1)*8*shankSep + shankSep*sI + elecPos(:,1), elecPos(:,2), 10, probeColor(probeNum,:), 'square' );
                    scatter((probeNum-1)*8*shankSep + shankSep*sI + chanPos(cc,1), chanPos(cc,2), 20, [.4 .2 .6], 'square', 'filled' );
                end
                xlim([-16,(numel(IMROfiles)-1)*8*shankSep + numel(IMROfiles)*3*shankSep+64]);
                ylim([-10,10000]);
                text((probeNum-1)*8*shankSep + 0.5*shankSep+64,10000,sprintf('Probe %d', probeNum))
                title(sprintf('%s, Protocol %s',days(d).name,regexprep(protocols(p).name,'_',' ')));
            end
        end
    end
    
    saveas(f,fullfile(basePath,'IMROWholeProtocol.png'),'png')
end
        
 
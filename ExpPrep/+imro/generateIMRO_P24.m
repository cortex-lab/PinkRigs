function fileName = generateIMRO_P24(patternTag, botRow, shankChoice, refElec, savePath)
% Build imro tables for some useful four shank combinations;
% also plot that selection.
%
% Output is saved in the directory where this script is run.
%
% patternTag = 'sin' all sites on "shankChoice" starting from "botRow", 0-448
% patternTag = 'hs4' 4-shank horizontal stripe of 96-channels starting from "botRow", valid values = 0-592
% patternTag = 'hs2' 2-shank horizontal stripe of 192-channels starting from "botRow", valid values = 0-496
% this code applies for phase 2 multishank probes only 
if ~exist('patternTag', 'var'); patternTag = 'hs4'; end
if ~exist('botRow', 'var'); botRow = 0; end
if ~exist('shankChoice', 'var'); shankChoice = 0; end     % 0-3, needed for patternType 0
if ~exist('refElec', 'var'); refElec = 0; end             % 0 for external, 1-4 for tip reference on shank 0-3
if ~exist('savePath', 'var'); savePath = '\\znas.cortexlab.net\code\PinkRigs\IMROFiles\'; end

shank = zeros(384,1,'single');
bank = zeros(384,1,'single');
elecInd = zeros(384,1,'single');
chans = zeros(384,1,'int32');

chans(:,1) = 0:383; % 384 channels
bMapOK = 1;

switch patternTag(1:3)
    
    case 'sin'
        %make a map with all sites on one shank, starting from electrode row n
        nameStr = sprintf( 'NPtype24_shank%d_botRow%d_ref%d', shankChoice, botRow, refElec );
        shank = shank + shankChoice;
        elecInd = botRow*2:(botRow*2 + 383);
        for i = 1:numel(elecInd)
            [bank(i), chans(i)] = ElecToChan( shank(i), elecInd(i) );
        end
        
    case 'hs4'
        %horizontal stripe of 2 channel blocks (96 sites) across all four shanks
        shElecInd = (botRow*2:(botRow*2 + 96)); %these are the electrode indices on each shank
        nameStr = sprintf( 'NPtype24_hStripe_botRow%d_ref%d', botRow, refElec );
        %loop over shanks; for each, calculate the channels that correspond
        %to these electrode indices       
        nE = 96; %electrodes in pattern per shank
        for sI = 0:3
            for i = 1:nE
                gEInd = sI*nE + i; % current electrode index for whole probe, plus one for MATLAB
                elecInd(gEInd) = shElecInd(i); % electrode index in whole selected set
                shank(gEInd) = sI;
                [bank(gEInd), chans(gEInd)] = ElecToChan( sI, elecInd(gEInd) );
                %fprintf("%d,%d,%d\n", bank(gEInd), chans(gEInd), elecInd(gEInd) );
            end
        end   
        
                
    case 'hs2'
        %horizontal stripe of 4 channel blocks (192 sites) across two shanks
        shElecInd = (botRow*2:(botRow*2 + 192)); %these are the electrode indices on each shank
        %loop over shanks; for each, calculate the channels that correspond
        %to these electrode indices
        nE = 192; %electrodes in pattern per shank
        shanks = [0 1];
        if exist('shankChoice', 'var') 
            shanks = shankChoice; 
        end
        nameStr = sprintf('NPtype24_2ShankhStripeSh%d-%d_botRow%d_ref%d', shanks(1), shanks(2), botRow, refElec );
        for sI = shanks
            for i = 1:nE
                gEInd = (sI-min(shanks))*nE + i; % current electrode index for whole probe, plus one for MATLAB
                elecInd(gEInd) = shElecInd(i); % electrode index in whole selected set
                shank(gEInd) = sI;
                [bank(gEInd), chans(gEInd)] = ElecToChan( sI, elecInd(gEInd) );
                %fprintf("%d,%d,%d\n", bank(gEInd), chans(gEInd), elecInd(gEInd) );
            end
        end
        
    otherwise
        fprintf('unknown pattern type\n');
        return;
end

%warn if there are duplicate channels
for i = 1:384
    if sum( chans(1:i-1)==chans(i) ) > 0
        fprintf( "duplicate channels => impossible map\n" );
        bMapOK = 0;
    end
end

if bMapOK
    %open a new file wherever we are
    fileLocation = savePath;
    fileName = fullfile(fileLocation,[nameStr,'.imro']);
    nmID = fopen(fileName,'w');

    [chans,sortI] = sort(chans);
    bank = int32(bank(sortI));
    shank = int32(shank(sortI));
    elecInd = elecInd(sortI);

    % imro table
    % print first entry, specifying probe type and number of channels
    fprintf(nmID,'(%d,%d)', 24, 384);
    for i = 1:numel(chans)
        fprintf(nmID,'(%d %d %d %d %d)', chans(i), shank(i), bank(i), refElec, elecInd(i) );
    end
    fprintf(nmID, '\n');

    fclose(nmID);

    % make a plot of all the electrode positions
    [~,~,~] = PlotElec24(shank, bank, chans, elecInd);
end

end

function [ chans, chanPos, chanShank ] = PlotElec24( shank, bank, chans, elecInd )

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
    chanShank = shank;
    
    % make a plot of all the electrode positions
    figure(1)
    shankSep = 250;
    for sI = 0:3
        cc = find(shank == sI);
        scatter( shankSep*sI + elecPos(:,1), elecPos(:,2), 30, 'k', 'square' ); hold on;
        scatter( shankSep*sI + chanPos(cc,1), chanPos(cc,2), 20, 'r', 'square', 'filled' ); hold on; 
    end
    xlim([-16,3*shankSep+64]);
    ylim([-10,10000]);
    title('NP2.0 MS shank view');
    hold off;
    
end

function [bank, chan] = ElecToChan( shank, elecInd )

%electrode to channel map
elecMap = zeros(4,8,'single');
elecMap(1,:) = [0,2,4,6,5,7,1,3];
elecMap(2,:) = [1,3,5,7,4,6,0,2];
elecMap(3,:) = [4,6,0,2,1,3,5,7];
elecMap(4,:) = [5,7,1,3,0,2,4,6];

bank = floor(elecInd/384);

%which block within the bank?
blockID = floor((elecInd - bank*384)/48);

%which channel within the block?
subBlockInd = mod((elecInd - bank*384), 48);

%get the channel number for that shank, bank, and block combo
chan = 48*elecMap(shank+1, blockID+1) + subBlockInd; 

end
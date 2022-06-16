function [chanPos, elecPos, shank] = getRecordingSites(binFileName,binFileFolder)

    metaData = readMetaData_spikeGLX(binFileName,binFileFolder);

    %% Extract info from metadata
    %%% Same as in plotIMROProtocol
    out = regexp(metaData.imroTbl,'\(|\)(|\)','split');
    out(1:2) = []; % empty + extra channel or something?
    out(end) = []; % empty

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
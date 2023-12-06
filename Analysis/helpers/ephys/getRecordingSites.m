function [chanPos, elecPos, shank, probeSN] = getRecordingSites(binFileName,binFileFolder)
    %% Fetches the recording site
    % WARNING: Should be double-checked for Neuropixels 1.0, or even for
    % 1-shank 2.0 probes.
    %
    % Parameters:
    % -------------------
    % binFileName: str
    %   Name of the bin file
    % binFileFolder: str
    %   Path to the bin file.
    %
    % Returns:
    % -------------------
    % chanPos: array
    %   Position of the channels on the probe
    % elecPos: array
    %   Position of the electrodes on the probe
    % shank: array
    %   Shank to which each channel (ordered as in chanPos) belongs
    % probeSN: int
    %   Serial number of the probe

    metaData = readMetaData_spikeGLX(binFileName,binFileFolder);
    probeSN = metaData.imDatPrb_sn;

    if strcmp(metaData.imDatPrb_type,'0')
        if isfield(metaData,'imDatPrb_port') && isfield(metaData,'imDatPrb_slot')
            probeType = '3B2';
        else 
            probeType = '3B1';
        end
    elseif strcmp(metaData.imDatPrb_type,'21')
        probeType = 'NP2.1';
    elseif strcmp(metaData.imDatPrb_type,'24')
        probeType = 'NP2.4';
    else
        error('Unrecognized probe type.')
    end

    if contains(probeType,'3B')
        vSep = 20;      % in um
        hSep = 32;
    else
        vSep = 15;      % in um
        hSep = 32;
    end

    % Get electrode positiosn
    % for 4 shank probes, pattern repeats for the four shanks?
    nElec = 1280;   % per shank; 

    elecPos = zeros(nElec, 2);

    elecPos(1:2:end,1) = 0;                %sites 0,2,4...
    elecPos(2:2:end,1) =  hSep;            %sites 1,3,5...

    % fill in y values
    viHalf = (0:(nElec/2-1))';                %row numbers
    elecPos(1:2:end,2) = viHalf * vSep;       %sites 0,2,4...
    elecPos(2:2:end,2) = elecPos(1:2:end,2);  %sites 1,3,5...

    if contains(probeType,'3B')
        elecPos(:,1) = elecPos(:,1) + mod(elecPos(:,2)/vSep,2)*hSep/2;
    end
    
    %% Extract info from metadata
    % 
    if isfield(metaData, 'snsShankMap') % old version of spike GLX
        out = regexp(metaData.snsShankMap,'\(|\)(|\)','split');
        out(1:2) = []; % empty + extra channel or something?
        out(end) = []; % empty

        shank = nan(1,numel(out));
        chanPos = nan(2,numel(out));
        for cc = 1:numel(out)
            feat = regexp(out{cc}, ':', 'split');
            shank(cc) = str2num(feat{1});
            chanPos(cc,1) = str2num(feat{2})*hSep;
            chanPos(cc,2) = str2num(feat{3})*vSep;
        end

        if contains(probeType,'3B')
            chanPos(:,1) = chanPos(:,1) + mod(chanPos(:,2)/vSep,2)*hSep/2;
        end

    elseif isfield(metaData, 'snsGeomMap') % new version of spike GLX
        out = regexp(metaData.snsGeomMap,'\(|\)(|\)','split');
        out(1:2) = []; % empty + extra channel or something?
        out(end) = []; % empty

        shank = nan(1,numel(out));
        chanPos = nan(2,numel(out));
        for cc = 1:numel(out)
            feat = regexp(out{cc}, ':', 'split');
            shank(cc) = str2num(feat{1});
            chanPos(cc,1) = str2num(feat{2});
            chanPos(cc,2) = str2num(feat{3});
        end

    end
    

%     %%% Same as in plotIMROProtocol
%     out = regexp(metaData.imroTbl,'\(|\)(|\)','split');
%     out(1:2) = []; % empty + extra channel or something?
%     out(end) = []; % empty
% 
%     chans = nan(1,numel(out));
%     shank = nan(1,numel(out));
%     bank = nan(1,numel(out));
%     elecInd = nan(1,numel(out));
%     for c = 1:numel(out)
%         chanProp = regexp(out{c},' ','split');
%         if numel(chanProp) == 5
%             if c == 1
%                 %%% Have a better way to check that?
%                 probeType = '2MS';
%             end
%             chans(c) = str2double(chanProp{1});
%             shank(c) = str2double(chanProp{2});
%             bank(c) = str2double(chanProp{3});
%             % 4 is refElec
%             elecInd(c) = str2double(chanProp{5});
%         elseif numel(chanProp) == 4
%             if c == 1
%                 warning('WATCH OUT: IMRO seems to be a Npx2.0 1 shank version--watch out, not sure what the chanMap is?')
%                 %%% Have a better way to check that?
%                 probeType = '2SS';
%             end
%             chans(c) = str2double(chanProp{1});
%             shank(c) = 0;
%             bank(c) = str2double(chanProp{2});
%             % 3 is refElec
%             elecInd(c) = str2double(chanProp{4});
%         elseif  numel(chanProp) == 6
%             if c == 1
%                 warning('WATCH OUT: IMRO seems to be a Npx 1.0. Just ouputting a gibberish hack?')
%                 %%% Have a bette way to check that?
%                 probeType = '1';
%             end
%             chans(c) = str2double(chanProp{1}); % ???
%             shank(c) = 0; % ???
%             bank(c) = str2double(chanProp{2}); % ???
%             elecInd(c) = c-1; % ???
%             % no idea what the others are
%         end
%     end

%     %% plot data -- taken from IMRO generation script
%     if contains(probeType,'NP')
%         % NP 2.0 MS (4 shank), probe type 24 electrode positions
%         nElec = 1280;   % per shank; pattern repeats for the four shanks
% 
%         elecPos = zeros(nElec, 2);
% 
%         elecPos(1:2:end,1) = 0;                %sites 0,2,4...
%         elecPos(2:2:end,1) =  hSep;            %sites 1,3,5...
% 
%         % fill in y values
%         viHalf = (0:(nElec/2-1))';                %row numbers
%         elecPos(1:2:end,2) = viHalf * vSep;       %sites 0,2,4...
%         elecPos(2:2:end,2) = elecPos(1:2:end,2);  %sites 1,3,5...
% 
%     elseif contains(probeType,'3B')
%         % NP 1.0
%         %%% VERY APPROXIMATE--whole thing is a hack
%         githubPath = fileparts(fileparts(which('autoRunOvernight.m')));
%         npx1_chanMap = load(fullfile(githubPath,'Processing','helpers','configFiles','neuropixPhase3B1_kilosortChanMap.mat'));
% 
%         elecPos = [npx1_chanMap.xcoords npx1_chanMap.ycoords];
%     end
%     chanPos = elecPos(elecInd+1,:);
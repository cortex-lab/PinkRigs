function [cids, cgs] = readClusterGroupsCSV(filename,curated)
    %% Gets the IDs and groups of the clusters.
    % Taken from the spikes repository.
    %
    % Parameters:
    % -------------------
    % filename: str
    %   Name of the file
    % curated: bool
    %   Whether the sorting has been curated
    %
    % Returns:
    % -------------------
    % cids 
    %   Is length nClusters, the cluster ID numbers
    % cgs 
    %   Is length nClusters, the "cluster group":
    %       - 0 = noise
    %       - 1 = mua
    %       - 2 = good
    %       - 3 = unsorted
    
    if ~exist('curated','var')
        curated = 0;
    end

    fid = fopen(filename);
    if curated
        % Not sure why, just don't want to mess up with things for now
        C = textscan(fid, '%s%s%s%s%s%s%s%s%s%s%s%s%s');
    else
        C = textscan(fid, '%s%s');
    end
    fclose(fid);
    
    cids = cellfun(@str2num, C{1}(2:end), 'uni', false);
    ise = cellfun(@isempty, cids);
    cids = [cids{~ise}];
    
    if curated
        idx = 9;
    else
        idx = 2;
    end
    isUns = cellfun(@(x)strcmp(x,'unsorted'),C{2}(2:end));
    isMUA = cellfun(@(x)strcmp(x,'mua'),C{2}(2:end));
    isGood = cellfun(@(x)strcmp(x,'good'),C{2}(2:end));
    cgs = zeros(size(cids));
    
    cgs(isMUA) = 1;
    cgs(isGood) = 2;
    cgs(isUns) = 3;
    
    cgs = uint8(cgs);
    cids = uint32(cids);
end
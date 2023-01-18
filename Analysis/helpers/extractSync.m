function extractSync(AP_filename, nChansTotal)
    %% Extracts the flipper from the ephys data.
    %
    % Parameters:
    % -------------------
    % AP_filename: str
    %   Name of the AP file
    % nChansTotal: int
    %   Total number of channels (usually 385)
.
    if ~exist('nChansTotal', 'var'); nChansTotal = 385; end
    
    d = dir(AP_filename);
    nSamps = d.bytes/2/nChansTotal;
    mmf = memmapfile(AP_filename,'Format',{'int16', [nChansTotal nSamps],'x'});
    
    disp('extracting sync...');
    sync = mmf.Data.x(385,:);    
    
    % save sync data
    save(fullfile(d.folder,'sync.mat'),'sync');
end

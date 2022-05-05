function extractSync(AP_filename, nChansTotal)
    %%% This function will extract the flipper from the ephys data.
    if ~exist('nChansTotal', 'var'); nChansTotal = 384; end
    
    d = dir(AP_filename);
    nSamps = d.bytes/2/nChansTotal;
    mmf = memmapfile(AP_filename,'Format',{'int16', [nChansTotal nSamps],'x'});
    
    disp('extracting sync...');
    sync = mmf.Data.x(385,:);    
    
    % save sync data
    save(fullfile(d.folder,'sync.mat'),'sync');
end

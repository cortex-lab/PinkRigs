function extractSync(AP_filename, nChansTotal)
    %% Extracts the flipper from the ephys data.
    %
    % Parameters:
    % -------------------
    % AP_filename: str
    %   Name of the AP file
    % nChansTotal: int
    %   Total number of channels (usually 385)

    if ~exist('nChansTotal', 'var'); nChansTotal = 385; end   

    if contains(AP_filename,'cbin')
       compressed_dat = 1; 
       disp('have to first decompress...'); 
       decompressPath = which('decompress_data.py');    
       d = dir(AP_filename);
       parent = dir(d.folder);
       parentfiles = {parent.name};
       is_ch = cellfun(@(x) contains(x,'ap.ch'),{parent.name});
       ch_file = parentfiles{is_ch}; 

       % perform decompression        
        [statusComp,resultComp] = system(['conda activate PinkRigs && ' ...
            'python ' decompressPath ' ' ...
             AP_filename ch_file ' && ' ...
            'conda deactivate']);


    else 
       compressed_dat = 0; 
    end 


    d = dir(AP_filename);

    nSamps = d.bytes/2/nChansTotal;
    mmf = memmapfile(AP_filename,'Format',{'int16', [nChansTotal nSamps],'x'});
    
    disp('extracting sync...');
    sync = mmf.Data.x(385,:);    
    
    % save sync data
    save(fullfile(d.folder,'sync.mat'),'sync');

    % recompress if it was compressed data

end

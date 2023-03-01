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
             AP_filename ' ' parent(1,1).folder '\' ch_file ' && ' ...
            'conda deactivate']);        

        % find new AP_binfile name
        parent = dir(d.folder);
        parentfiles = {parent.name};
        is_bin = cellfun(@(x) contains(x,'ap.bin'),{parent.name});
        AP_filename = [parent(1,1).folder '\' parentfiles{is_bin}]; 
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
    if compressed_dat==1
        disp('now  recompressing...')
        compressPath = which('compress_data.py');

        [statusComp,resultComp] = system(['conda activate PinkRigs && ' ...
            'python ' compressPath ' ' ...
             AP_filename ' && ' ...
            'conda deactivate']);
    end 

end

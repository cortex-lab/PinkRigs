function syncFT(AP_filename, nChansTotal, outputDir) 
d=dir(AP_filename);
nSamps = d.bytes/2/nChansTotal;
mmf = memmapfile(AP_filename,'Format',{'int16', [nChansTotal nSamps],'x'});

disp('extracting sync...'); 
sync=mmf.Data.x(385,:);    % save sync data---    
save(sprintf('%s\\sync.mat', outputDir),'sync');
end 

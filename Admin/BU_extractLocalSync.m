function extractLocalSync(localFolder)
    %%% This function will extract all the sync of the recordings saved
    %%% locally.

    localEphysFiles = dir(fullfile(localFolder, '/**/*.ap.bin'));
    localEphysFilesAgeInMins = (now-[localEphysFiles.datenum]')*24*60;
    localEphysFiles(localEphysFilesAgeInMins < 60) = []; 

    %%
    for i = 1:length(localEphysFiles)
        syncPath = fullfile(localEphysFiles(i).folder, 'sync.mat');
        if exist(syncPath, 'file'); continue; end
        metaS = readMetaData_spikeGLX(localEphysFiles(i).name,localEphysFiles(i).folder);

        apPath = fullfile(localEphysFiles(i).folder, localEphysFiles(i).name);
        fprintf('Couldn''t find the sync file for %s. Computing it.\n', ...
            localEphysFiles(i).name)
        extractSync(apPath, str2double(metaS.nSavedChans));
    end
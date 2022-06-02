function extractLocalSync(localFolder)
    %%% This function will extract all the sync of the recordings saved
    %%% locally.

    localEphysFiles = dir(fullfile(localFolder, '/**/*.ap.bin'));
    %%
    for i = 1:length(localEphysFiles)
        syncPath = fullfile(localEphysFiles(i).folder, 'sync.mat');
        if exist(syncPath, 'file'); continue; end
        metaS = readMetaData_spikeGLX(localEphysFiles(i).name,localEphysFiles(i).folder);

        apPath = fullfile(localEphysFiles(i).folder, localEphysFiles(i).name);
        fprintf('Couldn''t find the sync file for %s, %s. Computing it.\n', ...
            subjectFromBinName{i}, dateFromBinName{i})
        extractSync(apPath, str2double(metaS.nSavedChans));
    end
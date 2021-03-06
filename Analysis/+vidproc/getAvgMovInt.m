function getAvgMovInt(pathStub, nFramesToLoad)
    %%% This function will compute the average intensity of a movie, mainly
    %%% in order to extract the dark flashes.
    %%%
    %%% This code is inspired by the code from kilotrode
    %%% (https://github.com/cortex-lab/kilotrodeRig).
    
    if ~exist('nFramesToLoad', 'var')
        nFramesToLoad = [];
    end
    
    vr = VideoReader([pathStub '.mj2']);
    
    nF = get(vr, 'NumFrames');
    avgIntensity = zeros(1, nF);
    
    ROI = [1 1 vr.Width vr.Height];
    
    tic
    if isempty(nFramesToLoad) || isinf(nFramesToLoad)
        % load all
        chunkSize = 5000;
        numChunks = floor(nF/chunkSize);
        
        for ch = 1:numChunks
            img = read(vr, [(ch-1)*chunkSize+1 ch*chunkSize]);
            avgIntensity((ch-1)*chunkSize+1:ch*chunkSize) = squeeze(mean(mean(img(ROI(2):(ROI(2)+ROI(4)-1), ROI(1):(ROI(1)+ROI(3)-1),:,:), 1),2));
            fprintf(1, '%d / %d\n', ch*chunkSize, nF);
            toc
        end
        % if movie was shorter than chunkSize
        if isempty(ch)
            ch = 0;
        end
        % last bit
        img = read(vr, [ch*chunkSize+1 nF]);
        avgIntensity(ch*chunkSize+1:nF) = squeeze(mean(mean(img, 1),2));
        
        isLoaded = true(size(avgIntensity));
    else
        % load only a defined number of frames at the beginning and end of
        % the movie
        nFramesToLoad = min(nF,nFramesToLoad);
        img = read(vr, [1 nFramesToLoad]);
        avgIntensity(1:nFramesToLoad) = squeeze(mean(mean(img(ROI(2):(ROI(2)+ROI(4)-1), ROI(1):(ROI(1)+ROI(3)-1),:,:), 1),2));
        avgIntensity(nFramesToLoad+1:nF-nFramesToLoad) = avgIntensity(nFramesToLoad);
        
        img = read(vr, [nF-nFramesToLoad+1 nF]);
        avgIntensity(nF-nFramesToLoad+1:nF) = squeeze(mean(mean(img(ROI(2):(ROI(2)+ROI(4)-1), ROI(1):(ROI(1)+ROI(3)-1),:,:), 1),2));
        
        isLoaded = false(size(avgIntensity));
        isLoaded(1:nFramesToLoad) = true;
        isLoaded(nF-nFramesToLoad+1:nF) = true;
    end
    
    % delete and clear
    delete(vr)
    clear vr
    
    % save output
    save([pathStub '_avgIntensity.mat'], 'avgIntensity', 'isLoaded');
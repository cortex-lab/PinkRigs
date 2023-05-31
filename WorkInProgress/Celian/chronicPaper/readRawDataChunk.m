function data = readRawDataChunk(fileName,startTime,winSize,process)
    if nargin<4
        process = 0;
    end

    % Read data
    % reading .ch json file - contains info about compression
    chName = [fileName(1:end-4), 'ch'];
    fid = fopen(chName, 'r');
    data = fread(fid, 'uint8=>char');
    fclose(fid);
    cbinMeta = jsondecode(data');
    
    % build zmat info struct
    zmatInfo = struct;
    zmatInfo.type = cbinMeta.dtype;
    tmp = cast(1, cbinMeta.dtype);
    zmatInfo.byte = whos('tmp').bytes; % figuring out bytesPerSample programmatically
    zmatInfo.method = cbinMeta.algorithm;
    zmatInfo.status = 1;
    zmatInfo.level = cbinMeta.comp_level;
    
    % figuring out which chunks to read
    sampleStart = startTime*cbinMeta.sample_rate; % start time into the recording
    sampleEnd = sampleStart + winSize*cbinMeta.sample_rate; % window
    iChunkStart = find(sampleStart >= cbinMeta.chunk_bounds, 1, 'last');
    iChunkEnd = find(sampleEnd <= cbinMeta.chunk_bounds, 1, 'first') - 1;
    
    nSamplesPerChunk = diff(cbinMeta.chunk_bounds(iChunkStart:iChunkEnd+1));
    iSampleStart = max(sampleStart - cbinMeta.chunk_bounds(iChunkStart:iChunkEnd), 1);
    iSampleEnd = min(sampleEnd - cbinMeta.chunk_bounds(iChunkStart:iChunkEnd), nSamplesPerChunk);
    nChunks = iChunkEnd - iChunkStart + 1;
    
    nChannels = cbinMeta.n_channels;
    nSamples = cbinMeta.chunk_bounds([1:nChunks] + iChunkStart) - cbinMeta.chunk_bounds([1:nChunks] + iChunkStart - 1);
    chunkSizeBytes = cbinMeta.chunk_offsets([1:nChunks] + iChunkStart) - cbinMeta.chunk_offsets([1:nChunks] + iChunkStart - 1);
    offset = cbinMeta.chunk_offsets([1:nChunks] + iChunkStart - 1);
    allChannelIndices = 1:cbinMeta.n_channels;
    
    data = cell(nChunks, 1);
    for iChunk = 1:nChunks
        %     chunkInd = iChunk + iChunkStart - 1;
        % size of expected decompressed data for that chunk
        zmatLocalInfo = zmatInfo;
        zmatLocalInfo.size = [nSamples(iChunk)*nChannels, 1];
    
        % read a chunk from the compressed data
        fid = fopen(fileName, 'r');
        fseek(fid, offset(iChunk), 'bof');
        compData = fread(fid, chunkSizeBytes(iChunk), '*uint8');
        fclose(fid);
    
        decompData = zmat(compData, zmatLocalInfo);
        decompData = reshape(decompData, nSamples(iChunk), nChannels);
        chunkData = cumsum(decompData(:, allChannelIndices), 1);
        %     data(startIdx(iChunk):endIdx(iChunk), :) = chunkData(iSampleStart(iChunk):iSampleEnd(iChunk), :);
        data{iChunk} = chunkData(iSampleStart(iChunk):iSampleEnd(iChunk), :);
    end
    data = cell2mat(data);
    
    if process
        % Quick and dirty processing
        % Filter it
        for d = 1:size(data,2)
            data(:,d) = bandpass(double(data(:,d)),[400 9000],cbinMeta.sample_rate);
        end
        % CAR
        data = data - nanmedian(data,2);
    end
end
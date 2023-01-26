function [mua]=read_raw_data(AP_filename,ops,RID)

% ops must contain 
% ops.fsubsamp,ops.numChannels,ops.apSampleRate,ops.ichn 
% compute info that is the same independent of recording software 

recording_software=RID.recording_software; 
d=dir(AP_filename);
nSamps = d.bytes/2/ops.numChannels;
fsubsamp =ops.fsubsamp;
ichn=ops.ichn;
chunkSize = ops.apSampleRate * fsubsamp; % this number MUST be a multiple of fsubsamp
nChunksTotal = ceil(nSamps/chunkSize);
mua = zeros(ceil(nSamps/fsubsamp), size(ichn,2));

%newFs = ops.apSampleRate/fsubsamp;

if contains(recording_software,'OpenEphys')
    fid = fopen(apdataFilename, 'r');
    mua = zeros(ceil(nSamps/fsubsamp), size(ichn,2));
    chunkInd=0; 
    stepind=0; 
    while 1
        dat = fread(fid, [ops.numChannels chunkSize], '*int16');
        dat = double(dat);

    %     if chunkInd==2
    %         break
    %     end 

        if ~isempty(dat)
            % should pit the if chunkInd>2
            dat = double(permute(mean(reshape(dat(ichn, :), [size(ichn) size(dat,2)]),1), [3 2 1]));
            dat(fsubsamp * ceil(size(dat,1)/fsubsamp),:) = 0;

            % this is now binned into fsubsamp long bins 
            
            % should put switch case here for different types of dataset
            %mua0 = computeLFP_FT(dat,ops.apSampleRate, fsubsamp);
            mua0 = highpass_MUA(dat,ops.apSampleRate, fsubsamp);
            mua(stepind + (1:size(mua0,1)), :) = mua0;
            stepind = stepind+size(mua0,1);
        else 
            break
        end
        chunkInd = chunkInd+1;
        disp(chunkInd); 

    end 

    
elseif contains(recording_software,'SpikeGLX')
    mmf = memmapfile(AP_filename,'Format',{'int16', [ops.numChannels nSamps],'x'});

    for chunkID=1:nChunksTotal


        start=chunkSize*(chunkID-1)+1; 
        startdownsized=(start-1)/100+1; 

        if chunkID==nChunksTotal
            stop=nSamps; 
            stopdownsized=size(mua,1);
        else        
            stop=chunkSize*chunkID; 
            stopdownsized=stop/100;
        end


        dat = mmf.Data.x(1:384,start:stop);
        dat=double(dat); 

        dat = double(permute(mean(reshape(dat(ichn, :), [size(ichn) size(dat,2)]),1), [3 2 1]));
        dat(fsubsamp * ceil(size(dat,1)/fsubsamp),:) = 0;
        mua0 = computeLFP_FT(dat,ops.apSampleRate, fsubsamp);

        mua(startdownsized:stopdownsized,:)=mua0;

    end 
    
   sync=mmf.Data.x(385,:);    % save sync data---    
   save(sprintf('%s\\sync.mat', RID.path),'sync');
end 
end 


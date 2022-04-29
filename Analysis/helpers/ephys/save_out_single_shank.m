% read raw data and dump into .bin file according to shanks? 
path = 'Z:\AV008\2022-03-11\ephys\AV008_2022-03-11_ActivePassive_g0\AV008_2022-03-11_ActivePassive_g0_imec0';

AP_filename = [path '\AV008_2022-03-11_ActivePassive_g0_t0.imec0.ap.bin']; 
d=dir(AP_filename);
ops.numChannels=385; 
nSamps = d.bytes/2/ops.numChannels;
chunkSize = 1000000;
nChunksTotal = ceil(nSamps/chunkSize);

% read channelmap 
meta=readMetaData_spikeGLX(AP_filename,path); 
expression = '((\d+):(\d+):(\d+):(\d+))';%
shankdat=regexp(meta.snsShankMap,expression,'tokens');
%
shankID=zeros(size(shankdat,2),1); % determines which shank we are recording from 
for i=1:size(shankdat,2)
    currentband=shankdat{1,i}{1}; 
    currentband_dat=regexp(currentband,':','split');
    shankID(i)=str2double(currentband_dat{1});
end 

shank = 2;
% 
idx = find(shankID==shank);
mmf = memmapfile(AP_filename,'Format',{'int16', [ops.numChannels nSamps],'x'});

% read and write. have to do it in chunks.
outputFilename = [path '\' sprintf('shank_%.0d.ap.bin',shank)];
fidOut = fopen(outputFilename, 'w');
chunkInd = 1;
while 1

fprintf(1, 'chunk %d/%d\n', chunkInd, nChunksTotal);

dat = mmf.Data.x(idx,(chunkInd-1)*chunkSize+1:chunkInd*chunkSize);

if ~isempty(dat)

  fwrite(fidOut, dat, 'int16');

else
  break
end

chunkInd = chunkInd+1;
end
fclose(fidOut);


function RF_online(RID)

RID.probefolder=sprintf('%s_%s',RID.ephys_name,RID.probename);
RID.recording_software='SpikeGLX';%
add_packages(RID.Githubfolder);

%% get info for the raw data read-in
%compute channel map Phase1.0
%to do: get other probes channel map based on imro 
% [chanMap, ~, ~, ~, connected, ~] = makeForPRBimecP3(1);
% ichn = chanMap(connected>0);
% ichn = sort(ichn);
% ichn = reshape(ichn , 11, []); % 


if contains(RID.recording_software,'SpikeGLX')
    RID.path=[RID.ephys_folder '\' RID.probefolder '\'];
    Site_APband=[RID.path sprintf('%s_t0.%s.ap.bin',RID.ephys_name,RID.probename)];
    meta=readMetaData_spikeGLX(Site_APband, RID.path);
    RID.meta=meta;
    nChans = str2double(meta.nSavedChans);    
    [ichn,botrowID]=kilo.create_channelmapMultishank(Site_APband,RID.path,0);
    ichn = reshape(ichn , 8, []); % there will be 4 RFs for each bank 
elseif contains(recording_software,'OpenEphys')
    nChans=384;

end

ops.numChannels = nChans; % to edit for sure;
ops.apSampleRate = 30000;
ops.fsubsamp=100; 
ops.ichn=ichn;


%%
%% read in raw data and downsample 
% to do - compete different filtering cases
disp('reading in raw data...'); 
mua=read_raw_data(Site_APband,ops,RID);
disp('done.'); 

%% get sparsenoisedata aligned...
disp('alingning block to probe...'); 
expPath = fullfile(RID.root,sprintf('%.0f',RID.SparseNoise_expnum));
[sNdat]=sparseNoise_func(expPath,RID); 
disp('done.'); 

%%
% point is that the neural data is regularly sampled and it is from 0 
% sparseNoise is aligned to that 
% so one can index into the mua just by converting the indices to the right
% times 

newFs= ops.apSampleRate/ops.fsubsamp;
t_before=0.2; % 0.1 s before square onset
t_after=0.15; % # 0.1 s after square onset 
delay=0.05; 

before_ix=t_before*newFs;
after_ix=t_after*newFs; 
delay_ix=delay*newFs; 


xs=unique(sNdat.stimPositions(:,1)); 
ys=unique(sNdat.stimPositions(:,2));

RESPONSE=zeros(size(mua,2),size(ys,1),size(xs,1));
sNstartix = ceil(sNdat.stimTimesMain(:) * newFs)';
%
dt=-before_ix:1:after_ix;

for xct=1:size(xs,1)
     for yct=1:size(ys,1)         
         [ix,~]=find(sNdat.stimPositions(:,1)==xs(xct) & sNdat.stimPositions(:,2)==ys(yct));
         nrlix=sNstartix(ix);
         indref=nrlix+repmat(dt',1,numel(ix));
         resp_sq=reshape(mua(indref, :), [size(indref) size(mua,2)]);
         blFR=sum(resp_sq(1:before_ix,:,:),1)/t_before; 
         respFR=sum(resp_sq((before_ix+delay_ix):end,:,:),1)/(t_after-delay);
         rblsub=respFR-blFR;
         RESPONSE(:,yct,xct)=reshape(mean(rblsub,2),1,size(mua,2));
         
     end
end 

%%
figure;
for curr_depth=1:size(mua,2)
   rfMap=reshape(RESPONSE(curr_depth,:,:),size(ys,1),size(xs,1));
   subplot(size(mua,2),1,size(mua,2)-curr_depth+1);imagesc(rfMap'); axis image off; colormap parula;
   set(gca,'tag',sprintf('botrow %.0d-%.0d,use hStripe botrow %0.d',botrowID(curr_depth*8-7),botrowID(curr_depth*8),botrowID(curr_depth*8)-47))
   ax = gca;
   ax.Toolbar.Visible = 'off';
   %ylabel('192-195')
   %subplot(size(mua,2),1,curr_depth);imagesc(rfMap'); axis image off;
   
   %rfMap=reshape(RESPONSE(curr_depth,:,:),size(ys,1),size(xs,1));
   %figure; imagesc(rfMap',[-70000 70000]); colorbar;colormap spring;
   
   

end

clicksubplot;

%%
% rfMap=reshape(RESPONSE(9,:,:),size(ys,1),size(xs,1));
% figure; imagesc(rfMap');
% colorbar; colormap parula;




%%  util functions %% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
function add_packages(folderTools)
    %addpath(genpath(fullfile(folderTools, 'alyx-matlab')));
    addpath(genpath(fullfile(folderTools, 'npy-matlab')));
    addpath(genpath(fullfile(folderTools, 'kilotrodeRig')));
    addpath(genpath(fullfile(folderTools, 'spikes')));
    addpath(genpath(fullfile(folderTools, 'PinkRigs')));
    %addpath(genpath(fullfile(folderTools, 'AV_passive')));
end

%%
function  clicksubplot
while 1 == 1
    w = waitforbuttonpress;
      switch w 
          case 1 % keyboard 
              key = get(gcf,'currentcharacter'); 
              if key==27 % (the Esc key) 
                  try; delete(h); end
                  break
              end
          case 0 % mouse click 
              mousept = get(gca,'currentPoint');
              x = mousept(1,1);
              y = mousept(1,2);
              try; delete(h); end
              h = text(x,y,get(gca,'tag'),'vert','middle','horiz','center'); 
      end
end
end
end
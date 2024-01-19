clc; clear all;
extracted = loadOptoData('balanceTrials',0,'sepMice',1,'reExtract',1,'sepHemispheres',1,'sepDiffPowers',1); 

% for each set, fit the optoblock and then fit the control block
%%
for s=1:numel(extracted.subject)

    Block = extracted.data{s};
    controlBlock = filterStructRows(Block, ~Block.is_laserTrial);
    optoBlock = filterStructRows(Block, Block.is_laserTrial); 

    controlfit = plts.behaviour.GLMmulti(controlBlock, 'simpLogSplitVSplitA');
    controlfit.fit;

    ctrlbias(s) = controlfit.prmFits(1); 
    % refit opto with full
    %optoBlock.freeP = logical([1,1,1,0,1,1]);
    optoBlock.freeP = logical([1,0,0,0,0,0]);
    orifit = plts.behaviour.GLMmulti(optoBlock, 'simpLogSplitVSplitA');
    orifit.prmInit = controlfit.prmFits;
    orifit.fitCV(5); 
    deltaParams = mean(orifit.prmFits,1);
    dBias(s)=deltaParams(1);
    left_power(s) = optoBlock.stim_laser1_power(1);   
    right_power(s) = optoBlock.stim_laser2_power(1);   

end 

%%

f=figure; 
f.Position = [10,10,300,300];
plot(-[extracted.diff_power{:}],dBias,'.','MarkerEdgeColor','blue','MarkerSize',36);

xlabel('diff(laser power), mW')
ylabel('actual bias')
title(extracted.subject{1})
% calculate the additive delta_bias

%% 

for s=1:numel(dBias)

    if extracted.hemisphere{s}==0

        dL = dBias(([extracted.subject{:}]==extracted.subject{s}) & ([extracted.power{:}]==left_power(s)) & ([extracted.hemisphere{:}]==-1)); 
        dR = dBias(([extracted.subject{:}]==extracted.subject{s}) & ([extracted.power{:}]==right_power(s)) & ([extracted.hemisphere{:}]==1)); 
        dAdditive(s)= dL + dR; 
    else
        dAdditive(s) = dBias(s);
    end 

    power_texts{s} = sprintf('L:%.0f,R:%.0f mW',left_power(s),right_power(s));
end

%%

f=figure; 
f.Position = [10,10,400,400];
plot(dAdditive,dBias,'.','MarkerSize',36);
hold on
text(dAdditive,dBias+.2,power_texts,'FontSize',10);

hold on 
plot([-4,5],[-4,5],'k--')
xlabel('dL+dR')
ylabel('actual bias')
title(extracted.subject{1})

%%
% only plot the ones that are not 
isBi = (dAdditive-dBias)~=0; 

allSubjects = [extracted.subject{:}]; 
uniqueSubjects = unique(allSubjects);
c=['r','g','b']; 
figure; 
for s=1:numel(uniqueSubjects)
    currsel = isBi & (allSubjects==uniqueSubjects(s)); 
    scatter(dAdditive(currsel),dBias(currsel),100,c(s),'filled');
    hold on
end
hold on 
plot([-2.5,2.5],[-2.5,2.5],'k--')
plot([0,0],[-2.5,2.5],'k--')

xlabel('dL+dR')
ylabel('actual bias')

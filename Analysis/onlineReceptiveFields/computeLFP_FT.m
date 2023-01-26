
function lfp = computeLFP_FT(dat, Fs, fact)
%% adapted from Sylvia rfOnline
filtDat = dat;

% highPassCutoff = 300; % Hz
% [b1, a1] = butter(3, highPassCutoff/Fs, 'high');
% filtDat = filtfilt(b1,a1, filtDat);

lowPassCutoff = 300; % Hz
[b1, a1] = butter(5, lowPassCutoff/Fs, 'low');
filtDat = filtfilt(b1,a1, filtDat);

% filtDat = abs(filtDat);

NT = size(filtDat,1);
lfp = permute(mean(reshape(filtDat, fact, NT/fact, []), 1), [2 3 1]);

lfp = -lfp;
%%
% keyboard;
param = struct;
param.plotThis = 0;
% refractory period parameters
param.tauR = 0.0010; %refractory period time (s)
param.tauC = 0.0002; %censored period time (s)
param.maxRPVviolations = 0.5;
% percentage spikes missing parameters
param.maxPercSpikesMissing = 50;
param.computeTimeChunks = 0;
param.deltaTimeChunk = NaN;
% number of spikes
param.minNumSpikes = 300;
% waveform parameters
param.maxNPeaks = 2;
param.maxNTroughs = 1;
param.somatic = 1;
% amplitude parameters
param.nRawSpikesToExtract = 100;
param.minAmplitude = 15;
% recording parametrs
param.ephys_sample_rate = 30000;
param.nChannels = 385;
% distance metric parameters
param.computeDistanceMetrics = 0;
param.nChannelsIsoDist = NaN;
param.isoDmin = NaN;
param.lratioMin = NaN;
param.ssMin = NaN;
% plot
param.plotGlobal = 1;
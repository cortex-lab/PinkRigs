function unitType = bc_getQualityUnitType(qMetric,param)

    if param.computeDistanceMetrics && ~isnan(param.isoDmin)
        unitType = nan(length(qMetric.percSpikesMissing), 1);
        unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.somatic ~= param.somatic ...
            | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
            qMetric.waveformDuration > param.maxWvDuration | qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC
        unitType(any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing, 2)' & qMetric.nSpikes > param.minNumSpikes & ...
            any(qMetric.Fp <= param.maxRPVviolations, 2)' & ...
            qMetric.rawAmplitude > param.minAmplitude & qMetric.isoDmin >= param.isoDmin & isnan(unitType)) = 1; % SINGLE SEXY UNIT
        unitType(isnan(unitType)) = 2; % MULTI UNIT

    else
        unitType = nan(length(qMetric.percSpikesMissing), 1);
        unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.somatic ~= param.somatic ...
            | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
            qMetric.waveformDuration > param.maxWvDuration  | qMetric.waveformBaseline >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC
        unitType(any(qMetric.percSpikesMissing <= param.maxPercSpikesMissing, 2)' & qMetric.nSpikes > param.minNumSpikes & ...
            any(qMetric.Fp <= param.maxRPVviolations, 2)' & ...
            qMetric.rawAmplitude > param.minAmplitude & isnan(unitType)') = 1; % SINGLE SEXY UNIT
        unitType(isnan(unitType)') = 2; % MULTI UNIT
    end
end
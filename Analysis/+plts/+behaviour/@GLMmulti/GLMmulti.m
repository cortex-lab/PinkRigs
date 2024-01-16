classdef GLMmulti < matlab.mixin.Copyable
%% Class for multsensory GLM objects
%
% NOTE: this was based on earlier unisensory work by Peter Zatka-Haas
% NOTE: Typically called through "plts.behaviour.glmFit"
%
% Parameters: 
% ---------------
% inputBlockData (required): struct
%   Trial data for fitting
%
% modelString (default = []): string 
%   Indicates which model to fit. Different strings are interpreted in
%   the script "GLMMultiModels.m" In theory, this can be added after
%   creating the fitting object
%
% Returns: 
% -----------
% GLMmultt object with the following properties:
%   .modelString: modelString used
%   .prmLabels:   labels for the parameters used
%   .prmFits:     fitted values for paramters used
%   .prmBounds:   bounds used for fitting (not confidence interval)
%   .prmInit:     initial values for paramters used
%   .dataBlock:   behaviour data used for fitting (struct)
%   .pHat:        fitting information
%   .logLik:      logliklihood for final fit
%   .evalPoints:  points at which the curve was evaluated
%   .initGuess:   inital guess for values


    properties (Access=public)
        modelString;
        prmLabels;
        prmFits;
        prmBounds;
        prmInit;
        dataBlock;
        pHat;
        logLik;
        evalPoints;
        initGuess;
    end

    methods
        function obj = GLMmulti(inputBlockData, modelString)
            %% Input dataBlock must be a struct with fields: conditions and responseCalc
            inputBlockData.origMax = [max(abs(inputBlockData.stim_visDiff)) max(abs(inputBlockData.stim_audDiff))];
            inputBlockData.stim_visDiff = inputBlockData.stim_visDiff./max(inputBlockData.origMax(1),1e-15); % to avoid nans
            inputBlockData.stim_audDiff = inputBlockData.stim_audDiff./max(inputBlockData.origMax(2),1e-15);
            % Add previous choices and rewards
            inputBlockData.previous_respDirection = [0; inputBlockData.response_direction(1:end-1)];
            inputBlockData.previous_respFeedback = [0; inputBlockData.response_feedback(1:end-1)];

            obj.dataBlock = inputBlockData;
            obj.dataBlock.selectedTrials = ones(size(inputBlockData.stim_audDiff,1),1);
            tab = tabulate(obj.dataBlock.response_direction)/100;
            obj.initGuess = sum(tab(:,3).*log2(tab(:,3)));
            if exist('modelString', 'var')
                obj.GLMMultiModels(modelString);
            end
        end

        function fit(obj)
            %Non crossvalidated fitting
            if isempty(obj.modelString); error('Set model first'); end
            options = optimset('algorithm','interior-point','MaxFunEvals',100000,'MaxIter',2000, 'Display', 'none');

            mulIdx = obj.dataBlock.is_coherentTrial | obj.dataBlock.is_coherentTrial;
            if ~strcmpi(obj.modelString, 'simpLogSplitVSplitAUnisensory'); mulIdx = mulIdx*0; end
            backUpBlock = obj.dataBlock;
            obj.dataBlock = filterStructRows(obj.dataBlock, ~mulIdx);

            fittingObjective = @(b) (obj.calculateLogLik(b));
            [obj.prmFits,~,exitflag] = fmincon(fittingObjective, obj.prmInit, [], [], [], [], obj.prmBounds(1,:), obj.prmBounds(2,:), [], options);
            if ~any(exitflag == [1,2])
                obj.prmFits = nan(1,length(obj.prmLabels));
            end

            obj.dataBlock = backUpBlock;
            obj.pHat = obj.calculatepHat(obj.prmFits);
            obj.logLik = obj.calculateLogLik(obj.prmFits);
        end

        function fitCV(obj,nFolds)
            %Crossvalidated fitting
            if isempty(obj.modelString); error('Set model first'); end
            if ~exist('nFolds', 'var') || isempty(nFolds); nFolds = 5; end

            options = optimset('algorithm','interior-point','MaxFunEvals',100000,'MaxIter',2000, 'Display', 'none');
            cvObj = cvpartition(obj.dataBlock.response_direction,'KFold',nFolds);
            mulIdx = obj.dataBlock.is_coherentTrial | obj.dataBlock.is_conflictTrial;
            if ~strcmpi(obj.modelString, 'simpLogSplitVSplitAUnisensory'); mulIdx = mulIdx*0; end

            obj.prmFits = nan(cvObj.NumTestSets,length(obj.prmLabels));
            obj.pHat = [];
            obj.logLik = nan(cvObj.NumTestSets,1);
            for i = 1:cvObj.NumTestSets
                cvTrainObj = copy(obj);
                cvTrainObj.dataBlock = filterStructRows(cvTrainObj.dataBlock, cvObj.training(i) & ~mulIdx);
                disp(['Model: ' obj.modelString '. Fold: ' num2str(i) '/' num2str(cvObj.NumTestSets)]);

                fittingObjective = @(b) (cvTrainObj.calculateLogLik(b));
                [obj.prmFits(i,:),~,exitflag] = fmincon(fittingObjective, obj.prmInit(), [], [], [], [], obj.prmBounds(1,:), obj.prmBounds(2,:), [], options);
                if ~any(exitflag == [1,2]); obj.prmFits(i,:) = nan(1,length(obj.prmLabels)); end

                cvTestObj = copy(obj);
                cvTestObj.dataBlock = filterStructRows(cvTestObj.dataBlock, cvObj.test(i));
                pHatTested = cvTestObj.calculatepHat(obj.prmFits(i,:));
                obj.pHat(cvObj.test(i)) = pHatTested(sub2ind(size(pHatTested),(1:size(pHatTested,1))', cvTestObj.dataBlock.response_direction));
                obj.logLik(i) = -mean(log2(obj.pHat(cvObj.test(i))));
            end
        end


        function figureHand = plotFit(obj)
            if isempty(obj.prmFits); error('Model not fitted (non-crossvalidated) yet'); end
            params2use = mean(obj.prmFits,1);
            hold on;
            colorChoices = plts.general.selectRedBlueColors(obj.dataBlock.audValues);
            pHatCalculated = obj.calculatepHat(params2use,'eval');
            for audVal = obj.dataBlock.audValues(:)'
                plotIdx = obj.evalPoints(:,2)==audVal;
                plot(gca, obj.evalPoints(plotIdx,1), pHatCalculated(plotIdx,2), ...
                    'Color', colorChoices(obj.dataBlock.audValues==audVal,:), 'linewidth', 2);
            end
            maxContrast =obj.dataBlock.origMax(1);
            xlim([-1 1])
            set(gca, 'xTick', (-1):(1/4):1, 'xTickLabel', round(((-maxContrast):(maxContrast/4):maxContrast)*100));
            title({cell2mat(unique(obj.dataBlock.exp.subject)'); obj.modelString});
            hold off;
            figureHand = gca;
        end


        function h = plotParams(obj)
            if size(obj.prmFits,1)~=1; return; end
            bar(obj.prmFits);
            set(gca,'XTickLabel',obj.prmLabels,'XTick',1:numel(obj.prmLabels));
            title(obj.modelString);
            h=gca;
        end

        function pHatCalculated = calculatepHat(obj, P, tag)
            if isempty(obj.modelString); error('Set model first'); end
            if ~exist('tag', 'var'); tag = 'runModel'; end
            logOdds = obj.GLMMultiModels(tag, P);
            probRight = exp(logOdds)./(1+exp(logOdds));
            pHatCalculated = [1-probRight probRight];
        end


        function logLik = calculateLogLik(obj,testParams)
            pHatCalculated = obj.calculatepHat(testParams);
            responseCalc = obj.dataBlock.response_direction;
            logLik = -mean(log2(pHatCalculated(sub2ind(size(pHatCalculated),(1:size(pHatCalculated,1))', responseCalc))));
        end

    end
end
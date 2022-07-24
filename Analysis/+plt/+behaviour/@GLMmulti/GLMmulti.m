classdef GLMmulti < matlab.mixin.Copyable
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
            inputBlockData.stim_visDiff = inputBlockData.stim_visDiff./inputBlockData.origMax(1);
            inputBlockData.stim_audDiff = inputBlockData.stim_audDiff./inputBlockData.origMax(2);
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
            mulIdx = obj.dataBlock.tri.trialType.coherent | obj.dataBlock.tri.trialType.conflict;
            if ~strcmpi(obj.modelString, 'simpLogSplitVSplitAUnisensory'); mulIdx = mulIdx*0; end

            obj.prmFits = nan(cvObj.NumTestSets,length(obj.prmLabels));
            obj.pHat = [];
            obj.logLik = nan(cvObj.NumTestSets,1);
            for i = 1:cvObj.NumTestSets
                cvTrainObj = copy(obj); 
                cvTrainObj.dataBlock = prc.filtBlock(cvTrainObj.dataBlock, cvObj.training(i) & ~mulIdx);
                disp(['Model: ' obj.modelString '. Fold: ' num2str(i) '/' num2str(cvObj.NumTestSets)]);
                
                fittingObjective = @(b) (cvTrainObj.calculateLogLik(b));
                [obj.prmFits(i,:),~,exitflag] = fmincon(fittingObjective, obj.prmInit(), [], [], [], [], obj.prmBounds(1,:), obj.prmBounds(2,:), [], options);
                if ~any(exitflag == [1,2]); obj.prmFits(i,:) = nan(1,length(obj.prmLabels)); end
                
                cvTestObj = copy(obj); 
                cvTestObj.dataBlock = prc.filtBlock(cvTestObj.dataBlock, cvObj.test(i));
                pHatTested = cvTestObj.calculatepHat(obj.prmFits(i,:));
                obj.pHat(cvObj.test(i)) = pHatTested(sub2ind(size(pHatTested),(1:size(pHatTested,1))', cvTestObj.dataBlock.response_direction));
                obj.logLik(i) = -mean(log2(obj.pHat(cvObj.test(i))));
            end
        end
        
        function h = plotBlockData(obj)
            %%
            numTrials = prc.makeGrid(obj.dataBlock, ~isnan(obj.dataBlock.response_direction), @length, 1, 0, 1);
            numRightTurns = prc.makeGrid(obj.dataBlock, obj.dataBlock.response_direction==2, @sum, 1, 0, 1);
            
            audValues = [obj.dataBlock.audValues]./abs(max(obj.dataBlock.audValues));
            colorChoices = plt.selectRedBlueColors(audValues);
            
            [prob,confInterval] = arrayfun(@(x,z) binofit(x, z, 0.05), numRightTurns, numTrials, 'uni', 0);
            prob = cell2mat(cellfun(@(x) permute(x, [3,1,2]), prob, 'uni', 0));
            lowBound = cell2mat(cellfun(@(x) permute(x(:,1), [3,2,1]), confInterval, 'uni', 0));
            highBound = cell2mat(cellfun(@(x) permute(x(:,2), [3,2,1]), confInterval, 'uni', 0));
            grds = prc.getGridsFromBlock(obj.dataBlock);
            grds.visValues = grds.visValues./abs(max(grds.visValues(:)));
            for audVal = audValues(:)'
                idx = find(sign(grds.audValues)==audVal & numTrials>0);
                err = [prob(idx)-lowBound(idx), highBound(idx) - prob(idx)];
                errorbar(grds.visValues(idx),prob(idx),err(:,1),err(:,2),'.','MarkerSize',20, 'Color', colorChoices(audValues==audVal,:));
                hold on;
            end
            
            maxContrast =obj.dataBlock.origMax(1);
            xlim([-1 1])
            set(gca, 'xTick', (-1):(1/4):1, 'xTickLabel', round(((-maxContrast):(maxContrast/4):maxContrast)*100));
            
            xlabel('Contrast');
            ylabel('P( choice | contrast)');
            set(gca,'box','off');
            h=gca;
            set(gcf,'color','w');
        end
        
        function figureHand = plotFit(obj)
            if isempty(obj.prmFits); error('Model not fitted (non-crossvalidated) yet'); end
            params2use = mean(obj.prmFits,1);
            hold on;
            colorChoices = plt.selectRedBlueColors(obj.dataBlock.audValues);
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
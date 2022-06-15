classdef GLMmulti < matlab.mixin.Copyable
    properties (Access=public)
        modelString;
        prmLabels;
        prmFits;
        prmBounds;
        prmInit;
        blockData;
        pHat;
        logLik;
        evalPoints;
        initGuess;
    end
    
    methods
        function obj = GLMmulti(inputBlockData, modelString)
            %% Input blockData must be a struct with fields: conditions and responseCalc
            inputBlockData.origMax = [max(abs(inputBlockData.tri.stim.visDiff)) max(abs(inputBlockData.tri.stim.audDiff))];
            inputBlockData.tri.stim.visDiff = inputBlockData.tri.stim.visDiff./inputBlockData.origMax(1);
            inputBlockData.tri.stim.audDiff = inputBlockData.tri.stim.audDiff./inputBlockData.origMax(2);
            obj.blockData = inputBlockData;
            obj.blockData.tri.outcome.selectedTrials = ones(size(inputBlockData.tri.stim.audDiff,1),1);
            tab = tabulate(obj.blockData.tri.outcome.responseCalc)/100;
            obj.initGuess = sum(tab(:,3).*log2(tab(:,3)));
            if exist('modelString', 'var')
                obj.GLMMultiModels(modelString); 
            end
        end
        
        function fit(obj)
            %Non crossvalidated fitting
            if isempty(obj.modelString); error('Set model first'); end
            options = optimset('algorithm','interior-point','MaxFunEvals',100000,'MaxIter',2000, 'Display', 'none');
            
            mulIdx = obj.blockData.tri.trialType.coherent | obj.blockData.tri.trialType.conflict;
            if ~strcmpi(obj.modelString, 'simpLogSplitVSplitAUnisensory'); mulIdx = mulIdx*0; end
            backUpBlock = obj.blockData;
            obj.blockData = prc.filtBlock(obj.blockData, ~mulIdx);
            
            fittingObjective = @(b) (obj.calculateLogLik(b));
            [obj.prmFits,~,exitflag] = fmincon(fittingObjective, obj.prmInit, [], [], [], [], obj.prmBounds(1,:), obj.prmBounds(2,:), [], options);
            if ~any(exitflag == [1,2])
                obj.prmFits = nan(1,length(obj.prmLabels));
            end
            
            obj.blockData = backUpBlock;
            obj.pHat = obj.calculatepHat(obj.prmFits);
            obj.logLik = obj.calculateLogLik(obj.prmFits);
        end
        
        function fitCV(obj,nFolds)
            %Crossvalidated fitting
            if isempty(obj.modelString); error('Set model first'); end
            if ~exist('nFolds', 'var') || isempty(nFolds); nFolds = 5; end
            
            options = optimset('algorithm','interior-point','MaxFunEvals',100000,'MaxIter',2000, 'Display', 'none');
            cvObj = cvpartition(obj.blockData.tri.outcome.responseCalc,'KFold',nFolds);
            mulIdx = obj.blockData.tri.trialType.coherent | obj.blockData.tri.trialType.conflict;
            if ~strcmpi(obj.modelString, 'simpLogSplitVSplitAUnisensory'); mulIdx = mulIdx*0; end

            obj.prmFits = nan(cvObj.NumTestSets,length(obj.prmLabels));
            obj.pHat = [];
            obj.logLik = nan(cvObj.NumTestSets,1);
            for i = 1:cvObj.NumTestSets
                cvTrainObj = copy(obj); 
                cvTrainObj.blockData = prc.filtBlock(cvTrainObj.blockData, cvObj.training(i) & ~mulIdx);
                disp(['Model: ' obj.modelString '. Fold: ' num2str(i) '/' num2str(cvObj.NumTestSets)]);
                
                fittingObjective = @(b) (cvTrainObj.calculateLogLik(b));
                [obj.prmFits(i,:),~,exitflag] = fmincon(fittingObjective, obj.prmInit(), [], [], [], [], obj.prmBounds(1,:), obj.prmBounds(2,:), [], options);
                if ~any(exitflag == [1,2]); obj.prmFits(i,:) = nan(1,length(obj.prmLabels)); end
                
                cvTestObj = copy(obj); 
                cvTestObj.blockData = prc.filtBlock(cvTestObj.blockData, cvObj.test(i));
                pHatTested = cvTestObj.calculatepHat(obj.prmFits(i,:));
                obj.pHat(cvObj.test(i)) = pHatTested(sub2ind(size(pHatTested),(1:size(pHatTested,1))', cvTestObj.blockData.tri.outcome.responseCalc));
                obj.logLik(i) = -mean(log2(obj.pHat(cvObj.test(i))));
            end
        end
        
        function h = plotBlockData(obj)
            %%
            numTrials = prc.makeGrid(obj.blockData, ~isnan(obj.blockData.tri.outcome.responseCalc), @length, 1, 0, 1);
            numRightTurns = prc.makeGrid(obj.blockData, obj.blockData.tri.outcome.responseCalc==2, @sum, 1, 0, 1);
            
            audValues = [obj.blockData.audValues]./abs(max(obj.blockData.audValues));
            colorChoices = plt.selectRedBlueColors(audValues);
            
            [prob,confInterval] = arrayfun(@(x,z) binofit(x, z, 0.05), numRightTurns, numTrials, 'uni', 0);
            prob = cell2mat(cellfun(@(x) permute(x, [3,1,2]), prob, 'uni', 0));
            lowBound = cell2mat(cellfun(@(x) permute(x(:,1), [3,2,1]), confInterval, 'uni', 0));
            highBound = cell2mat(cellfun(@(x) permute(x(:,2), [3,2,1]), confInterval, 'uni', 0));
            grds = prc.getGridsFromBlock(obj.blockData);
            grds.visValues = grds.visValues./abs(max(grds.visValues(:)));
            for audVal = audValues(:)'
                idx = find(sign(grds.audValues)==audVal & numTrials>0);
                err = [prob(idx)-lowBound(idx), highBound(idx) - prob(idx)];
                errorbar(grds.visValues(idx),prob(idx),err(:,1),err(:,2),'.','MarkerSize',20, 'Color', colorChoices(audValues==audVal,:));
                hold on;
            end
            
            maxContrast =obj.blockData.origMax(1);
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
            colorChoices = plt.selectRedBlueColors(obj.blockData.audValues);
            pHatCalculated = obj.calculatepHat(params2use,'eval');
            for audVal = obj.blockData.audValues(:)'
                plotIdx = obj.evalPoints(:,2)==audVal;
                plot(gca, obj.evalPoints(plotIdx,1), pHatCalculated(plotIdx,2), ...
                    'Color', colorChoices(obj.blockData.audValues==audVal,:), 'linewidth', 2);
            end
            maxContrast =obj.blockData.origMax(1);
            xlim([-1 1])
            set(gca, 'xTick', (-1):(1/4):1, 'xTickLabel', round(((-maxContrast):(maxContrast/4):maxContrast)*100));
            title({cell2mat(unique(obj.blockData.exp.subject)'); obj.modelString});
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
            responseCalc = obj.blockData.tri.outcome.responseCalc;
            logLik = -mean(log2(pHatCalculated(sub2ind(size(pHatCalculated),(1:size(pHatCalculated,1))', responseCalc))));
        end
        
    end
end
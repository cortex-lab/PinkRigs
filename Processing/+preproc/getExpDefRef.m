function expDefRef = getExpDefRef(expDef)
    %% Gets the exp def reference file name for each specific expDef.
    %
    % Parameters:
    % -------------------
    % expDef: str
    %   ExpDef name
    %
    % Returns: 
    % -------------------
    % expDefRef: str
    %   Associated expDef reference from +expDef package
            
    if contains(expDef,'imageWorld')
        expDefRef = 'imageWorld';
        
    elseif contains(expDef,'spontaneousActivity')
        expDefRef = 'spontaneous';
        
    elseif contains(expDef,'sparseNoise')
        expDefRef = 'sparseNoise';
         
    elseif contains(expDef,'multiSpaceWorld_checker_training') || ...
            contains(expDef, 'multiSpaceWorld_checker') || ...
            contains(expDef,'multiSpaceSwitchWorld')
        expDefRef = 'multiSpaceTraining';
        
    elseif strcmpi(expDef,'multiSpaceWorld') || strcmpi(expDef,'multiSpaceWorldNewNames')
        expDefRef = 'multiSpaceWorld';

    elseif contains(expDef,'postactive')
        expDefRef = 'AVPassive';

    elseif contains(expDef,'extended') || contains(expDef,'spatialIntegrationFlora') || contains(expDef,'updatechecker')
        expDefRef = 'AVPassive_extended';

    elseif contains(expDef,'Vid')
        expDefRef = 'spontaneous';

    else
        error('ExpDef reference does not exist for expDef %s',expDef)
    end

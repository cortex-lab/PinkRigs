
githubPath = fileparts(fileparts(which('autoRunOvernight.m')));

fprintf('Running "runFacemap" ... \n')
% update environment 
eveningFacemapPath = [githubPath '\Analysis\+vidproc\_facemap\run_facemap.py'];

facemapEnvFilePath = [githubPath '\facemap_environment.yaml'];

% Original 
eveningFacemapPath = [githubPath '\Analysis\\+vidproc\_facemap\run_facemap.py'];
[statusFacemap resultFacemap] = system(['activate facemap && ' ...
     'cd ' githubPath ' && ' ...
    'conda env update --file facemap_environment.yaml --prune' ' &&' ...
    'python ' eveningFacemapPath ' &&' ...
    'conda deactivate']);

% Test 
[statusFacemap resultFacemap] = system(['conda activate facemap && ' ...
    'conda env update --file ' facemapEnvFilePath ' --prune' ' &&' ...
    'python ' eveningFacemapPath ' &&' ...
    'conda deactivate']);


if statusFacemap > 0
    fprintf('Facemap failed with error "%s".\n', resultFacemap)
end

disp(resultFacemap);
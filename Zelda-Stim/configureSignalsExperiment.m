function e = configureSignalsExperiment(paramStruct, rig)
%exp.configureSignalsExperiment Setup Signals Experiment class
%   Instantiate the exp.SignalsExp class and configure the object.
%   Subclasses may be instantiated here using the type parameter.
%
%   Inputs:
%     paramStruct : a SignalsExp parameter structure
%     rig : a structure of rig hardware objects returned by hw.devices)
%
%   Output:
%     e : a SignalsExp object
%
%   Example:
%     rig = hw.devices;
%     rig.stimWindow.open();
%     pars = exp.inferParameters(@choiceWorld);
%     e = exp.configureSignalsExperiment(pars, rig);
%     e.run([]);
%
% See also exp.configureFilmExperiment, exp.configureChoiceExperiment

%% Create the experiment object
e = exp.SignalsExpNoVis(paramStruct, rig);
e.Type = 'signals no vis'; %record the experiment type

end


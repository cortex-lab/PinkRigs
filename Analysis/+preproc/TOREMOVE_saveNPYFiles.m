function saveNPYFiles(s,savePath)
    %% Will save the content of s in the ONE format (object.property.extension)
    %
    % Parameters:
    % -------------------
    % s: struct
    %   Sound waveform

    % Check it's a structure
    if ~isstruct(s)
        error('Are you sure this is the right input? It''s a %s', class(s))
    end
    
    % Create path
    if ~exist(savePath,'dir')
        mkdir(savePath)
    end
        
    % Loop through all the fields to save them individually
    fieldNamesObjects = fieldnames(s);
    for f = 1:numel(fieldNamesObjects)
        obj = fieldNamesObjects{f};
        fieldNamesProperties = fieldnames(s.(obj));
        for ff = 1:numel(fieldNamesProperties)
            prop = fieldNamesProperties{ff};
            var = [s.(obj).(prop)];
            filename = fullfile(savePath,sprintf('%s.%s.npy',obj,prop));
            
            % save the NPY
            if ~isstruct(var) && ~iscell(var)
                writeNPY(var, filename)
            else
                if isstruct(var)
                    % is a struct (rawWaveforms)--split it
                    fieldNamesSubprop = fieldnames(var);
                    for fff = 1:numel(fieldNamesSubprop)
                        subprop = fieldNamesSubprop{fff};
                        subvar = squeeze(cat(3,var.(subprop)));
                        subfilename = fullfile(savePath,sprintf('%s.%s_%s.npy',obj,prop,subprop));
                        writeNPY(subvar, subfilename)
                    end
                elseif iscell(var)
                    % save it as a .mat for now... 
                    save(regexprep(filename,'.npy','.mat'),'var');
                end
            end
        end
    end
end
function filePath = saveONEFormat(var,savePath,objectName,attributeName,extensionName,stub)
    %%% Write a file with the ONE format.
    
    if ~exist('stub','var')
       stub = '';
    else
        if ~strcmp(stub(1),'.')
            stub = ['.' stub];
        end
    end

    filePath = fullfile(savePath,sprintf('%s.%s%s.%s',objectName,attributeName,stub,extensionName));

    switch extensionName
        case 'npy'
            writeNPY(var, filePath);
        case {'pqt','parquet'}
            if istype(var,'struct')
                var = struct2table(var);
            end
            parquetwrite(filePath, var);
        otherwise
            error('Sorry, can''t find this extension: %s.',subfilename)
    end

end
function filePath = saveONEFormat(var,savePath,objectName,attributeName,extensionName,stub)
    %% Writes a file in ONE format
    %
    % Parameters:
    % -------------------
    % var: array, table, etc.
    %   Variable to save.
    % savePath: str
    %   Path
    % objectName: str
    %   Name of the object (e.g., 'spikes')
    % attributeName: str
    %   Name of the attribute (e.g., 'times')
    % extensionName: str
    %   Extension (e.g., '.npy')
    % stub (optional): str
    %   Little string to add to the file name

    if ~exist('stub','var')
        stub = '';
    else
        if ~strcmp(stub(1),'.')
            stub = ['.' stub];
        end
    end

    filePath = fullfile(savePath,sprintf('%s.%s%s.%s',objectName,attributeName,stub,extensionName));
    filePathLarge = fullfile(savePath,sprintf('%s.%s%s.%s',objectName,[attributeName '_largeData'],stub,extensionName));

    switch extensionName
        case 'npy'
            writeNPY(var, filePath);
        case {'pqt','parquet'}
            if isstruct(var)

                largeVar = struct;
                for currField = fields(var)'
                    if iscell(var.(currField{1}))
                        largeVar.(currField{1}) = var.(currField{1});
                        var = rmfield(var, currField{1});
                    end
                end

                Sbytes = whosstruct(var);
                totalSize = sum(structfun(@(x) x, Sbytes));
                fracSize =  structfun(@(x) x, Sbytes)/totalSize;
                while length(fracSize) > 4 && sum(maxk(fracSize,floor(0.2*length(fracSize))))>0.75
                    varFields = fields(var);
                    maxField = varFields(fracSize == max(fracSize));
                    largeVar.(maxField{1}) = var.(maxField{1});
                    var = rmfield(var, maxField{1});

                    Sbytes = whosstruct(var);
                    totalSize = sum(structfun(@(x) x, Sbytes));
                    fracSize =  structfun(@(x) x, Sbytes)/totalSize;
                end

                var = struct2table(var);
                largeVar = struct2table(largeVar);
            else 
                largeVar = [];
            end

            parquetwrite(filePath, var);
            if ~isempty(largeVar)
                parquetwrite(filePathLarge, largeVar);
            end
        otherwise
            error('Sorry, can''t find this extension: %s.',subfilename)
    end

end

%% Funtion to check size of each struct field
function Sbytes = whosstruct(S)
    for fld = fieldnames(S)'
        val = S.(fld{1});
        tmp = whos('val');
        Sbytes.(fld{1}) = tmp.bytes;
    end
end
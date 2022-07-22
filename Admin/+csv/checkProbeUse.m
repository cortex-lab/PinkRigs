function [subjects, implantDate, explantDate] = checkProbeUse(probeSerial,select,plt)
    %%% Will give past info about this probe
    %%% probeSN can be a cell array
 
    if ~exist('select','var')
        select = 'all';
    end
    
    if ~exist('plt','var')
        plt = 0;
    end
    
    csvData = csv.readTable(csv.getLocation('main'));
    csvFields = fields(csvData);
    serialsFromCSV = cellfun(@(x) csvData.(x), csvFields(contains(csvFields, 'serial')), 'uni', 0)';
    serialsFromCSV = cell2mat(cellfun(@str2double, serialsFromCSV, 'uni', 0));
    
    if ~exist('probeSerial','var') || isempty(probeSerial)
        probeSerial = unique(serialsFromCSV);
        probeSerial(isnan(probeSerial)) = [];
    end     
    if isnumeric(probeSerial); probeSerial = num2cell(probeSerial); end
   
    subjects = cell(1,numel(probeSerial));
    implantDate = cell(1,numel(probeSerial));
    explantDate = cell(1,numel(probeSerial));
    for psn = 1:numel(probeSerial)  
        % get their orders and dates
        snIdx = find(serialsFromCSV == probeSerial{psn});
        [subjectIdx,probeIdx] = ind2sub(size(serialsFromCSV),snIdx);
        if ~isempty(subjectIdx)
            for pp = 1:numel(probeIdx)
                subjects{psn}{pp} = csvData.Subject{subjectIdx(pp)};
                implantDate{psn}{pp} = csvData.(sprintf('P%d_implantDate',probeIdx(pp)-1)){subjectIdx(pp)};
                explantDate{psn}{pp} = csvData.(sprintf('P%d_explantDate',probeIdx(pp)-1)){subjectIdx(pp)};
            end
            undefIdx = cellfun(@isempty, explantDate{psn}) | ...
                cellfun(@(x) strcmp(x,'Permanent'), explantDate{psn});
            explantDate{psn}(undefIdx) = {datestr(now+20,'yyyy-mm-dd')};
            
            % resort it
            [~,sortIdx] = sort(datenum(explantDate{psn},'yyyy-mm-dd'),'ascend');
            subjects{psn} = subjects{psn}(sortIdx);
            implantDate{psn} = implantDate{psn}(sortIdx);
            explantDate{psn} = explantDate{psn}(sortIdx);
        else
            fprintf('No valid entry found for serial number: %d \n', probeSerial{psn})
        end
    end
    
    %% Plot figure
    if plt  
        figure;
        for psn = 1:numel(probeSerial)
            subplot(ceil(sqrt(numel(probeSerial))),ceil(sqrt(numel(probeSerial))),psn); hold all
            minDate = datenum(implantDate{psn}{1},'yyyy-mm-dd');
            maxDate = datenum(explantDate{psn}{end},'yyyy-mm-dd');
            miceMat = zeros(numel(subjects{psn}),maxDate-minDate+1);
            for pp = 1:numel(subjects{psn})
                miceMat(pp,datenum(implantDate{psn}{pp},'yyyy-mm-dd')-minDate+1:datenum(explantDate{psn}{pp},'yyyy-mm-dd')-minDate+1) = pp;
            end
            imagesc(miceMat);
            colormap('RedBlue')
            caxis([-max(cellfun(@numel, subjects)) max(cellfun(@numel, subjects))])
            
            % add animal names
            yticks(1:numel(subjects{psn}))
            yticklabels(subjects{psn})
            
            % put proper implant/explant dates along x axis
            dates = [datenum(implantDate{psn},'yyyy-mm-dd')-minDate+1; ...
                datenum(explantDate{psn},'yyyy-mm-dd')-minDate];
            dateLabels = [implantDate{psn} explantDate{psn}];
            [~,sortDatesIdx] = sort(dates);
            xticks(dates(sortDatesIdx))
            xticklabels(dateLabels(sortDatesIdx))
            xtickangle(45)
            
            % SN in title
            title(sprintf('Probe #%d',probeSerial{psn}))
        end
    end

    %% Subselect for output if wanted
    
    switch select
        case 'all'
            % do nothing
        case 'last'
            for psn = 1:numel(probeSerial) 
                subjects{psn} = subjects{psn}(end);
                implantDate{psn} = implantDate{psn}(end);
                explantDate{psn} = explantDate{psn}(end);
            end
        case 'current'
            % put nan when the probe has been explanted
            for psn = 1:numel(probeSerial)
                if datenum(explantDate{psn}(end))>now
                    subjects{psn} = subjects{psn}(end);
                    implantDate{psn} = implantDate{psn}(end);
                    explantDate{psn} = explantDate{psn}(end);
                else
                    subjects{psn} = [];
                    implantDate{psn} = [];
                    explantDate{psn} = [];
                end
            end
        otherwise
            error('Selection not understood.')
    end
end
function loadNonZeldaSubject(subjectList)
    %% Get this subject's recordings
    
    serverLocations = getServersList;
    D = {};
    for subject = subjectList'
        for server = serverLocations'
            d = dir(fullfile(server{1}, subject{1},'**','*ap.*bin'));
    
            ready = false(1,numel(d));
            for dd = 1:numel(d)
                if exist(fullfile(d(dd).folder,'pyKS\output\ibl_format'))
                    ready(dd) = true;
                end
            end

            D = [D d];
        end
    end
    D = cat(1,D{:});

    %% Get 
    for ks = 1:KSFolderList
        % Get the spike and cluster info
        spkONE = preproc.getSpikeDataONE(KSFolder);

        % Get IBL quality metrics
        qMetrics = preproc.getQMetrics(KSFolder,'IBL');
        % The qMetrics don't get calculated for some trash units, but we need to keep
        % the dimensions consistent of course...
        qMetrics = removevars(qMetrics,{'cluster_id_1'}); % remove a useless variable
        cname = setdiff(spkONE.clusters.av_IDs,qMetrics.cluster_id);

        added_array = qMetrics(1,:);
        added_array{1,added_array.Properties.VariableNames(1:end-1)} = nan;
        added_array{1,'ks2_label'} = {'noise'};

        for i=1:numel(cname)
            added_array(1,'cluster_id') = {cname(i)};
            qMetrics = [qMetrics;added_array];
        end
        qMetrics = sortrows(qMetrics,'cluster_id');

        % Get Bombcell metrics
        if exist(fullfile(KSFolder,'qMetrics','templates._bc_qMetrics.parquet'),'file')
            qMetrics = preproc.getQMetrics(KSFolder,'bombcell');
            qMetrics.clusterID = qMetrics.clusterID - 1; % base 0 indexing
            saveONEFormat(qMetrics, ...
                probeONEFolder,'clusters','_bc_qualityMetrics','pqt',stub);
        else
            error('Quality metrics haven''t been computed...')
        end
    end
end
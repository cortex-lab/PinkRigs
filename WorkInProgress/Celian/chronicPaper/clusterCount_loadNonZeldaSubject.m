function [clusterNum, recLocAll, days, expInfoAll] = clusterCount_loadNonZeldaSubject(subjectList,getQM,getPos)
    %% Get this subject's recordings
    
    serverLocations = getServersList;
    D = {};
    for subject = subjectList
        for server = serverLocations'
            d = dir(fullfile(server{1}, subject{1},'**','*ap.*bin'));
    
            ready = false(1,numel(d));
            for dd = 1:numel(d)
                if exist(fullfile(d(dd).folder,'pyKS\output\ibl_format'))
                    ready(dd) = true;
                end
            end

            D = [D d(ready)];
        end
    end
    D = cat(1,D{:});

    KSFolderList = cellfun(@(x) fullfile(x, 'pyKS\output\'), {D.folder}, 'uni', 0);

    % Implantation date?
    implantDates = cell(1,numel(subjectList));
    for ss = 1:numel(subjectList)
        switch subjectList{ss}
            case 'EB014'
                implantDates{ss} = '2022-04-26';
            case 'EB019'
                implantDates{ss} = '2022-06-29';
            case 'CB015'
                implantDates{ss} = '2021-09-09';
            case 'Churchland001'
                implantDates{ss} = '2022-10-27';
        end
    end

    %% Get all info from 

    clusterNum = nan(1,numel(KSFolderList));
    days = nan(1,numel(KSFolderList));
    recLocAll = cell(1,numel(KSFolderList));
    expInfoAll = cell(1,numel(KSFolderList));
    for nn = 1:numel(KSFolderList)
        KSFolder = KSFolderList{nn};
        fprintf('Loading data from folder %s...', KSFolder)

        % Get recording location
        binFile = D(nn);
        [chanPos,~,shanks,probeSN] = getRecordingSites(binFile(1).name,binFile(1).folder);
        shankIDs = unique(shanks);
        botRow = min(chanPos(:,2));

        % Get info
        sp = split(binFile(1).folder,'\');
        subject = sp{5};
        expDate = sp{6};
        implantDate = implantDates(contains(subjectList,subject));
        
        % Build tags etc
        days(nn) = datenum(expDate);
        days(nn) = days(nn)-datenum(implantDate);
        recLocAll{nn} = [subject '__' num2str(probeSN) '__' num2str(shankIDs) '__' num2str(botRow)];

        expInfoAll{nn}.daysSinceImplant = days(nn);

        attr = {'_av_KSLabels','_av_IDs'};
        if getQM
            attr = cat(2,attr,{'qualityMetrics'});
        end
        if getPos
            attr = cat(2,attr,{'_av_xpos','depths'});
        end

        % Get the spike and cluster info
        spkONE = preproc.getSpikeDataONE(KSFolder);
        clusters.av_IDs = spkONE.clusters.av_IDs;
        clusters.av_KSLabels = spkONE.clusters.av_KSLabels;

        % Get cluster count
        goodUnits = clusters.av_KSLabels == 2;
        clusterNum(nn) = sum(goodUnits);

        if getPos
            clusters.depths = spkONE.clusters.depths;
            clusters.av_xpos = spkONE.clusters.av_xpos;
        end

        if getQM
            % Get IBL quality metrics
            clusters.qualityMetrics = preproc.getQMetrics(KSFolder,'IBL');
            % The qMetrics don't get calculated for some trash units, but we need to keep
            % the dimensions consistent of course...
            clusters.qualityMetrics = removevars(clusters.qualityMetrics,{'cluster_id_1'}); % remove a useless variable
            cname = setdiff(spkONE.clusters.av_IDs,clusters.qualityMetrics.cluster_id);

            added_array = clusters.qualityMetrics(1,:);
            added_array{1,added_array.Properties.VariableNames(1:end-1)} = nan;
            added_array{1,'ks2_label'} = {'noise'};

            for i=1:numel(cname)
                added_array(1,'cluster_id') = {cname(i)};
                clusters.qualityMetrics = [clusters.qualityMetrics;added_array];
            end
            clusters.qualityMetrics = sortrows(clusters.qualityMetrics,'cluster_id');

            % Get Bombcell metrics
            clusters.bc_qualityMetrics = preproc.getQMetrics(KSFolder,'bombcell');
            clusters.bc_qualityMetrics.clusterID = clusters.bc_qualityMetrics.clusterID - 1; % base 0 indexing
        end

        % Save in expInfoAll
        expInfoAll{nn}.dataSpikes{1}.probe0.clusters = clusters;

        expInfoAll{nn} = struct2table(expInfoAll{nn});
        fprintf('Done.\n')
    end
end
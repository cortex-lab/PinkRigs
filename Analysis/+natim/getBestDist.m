function [BestDist,distPair] = getBestDist(BestMatch, XPos, Depth)

    distPair = cell(size(BestMatch,1),size(BestMatch,1));
    for d1 = 1:size(BestMatch,1)
        nCd1 = size(XPos{d1},1);
        for d2 = 1:size(BestMatch,2)
            nCd2 = size(XPos{d2},1);
            distPair{d1,d2} = sqrt((repmat(XPos{d1},[1,nCd2]) - repmat(XPos{d2}',[nCd1,1])).^2 + ...
                (repmat(Depth{d1},[1,nCd2]) - repmat(Depth{d2}',[nCd1,1])).^2);
            
            % get BestDist -- not optimal
            for i = 1:nCd1
                isNotNan = ~isnan(BestMatch{d1,d2}(i,:));
                BestDist{d1,d2}(i,isNotNan) = distPair{d1,d2}(i,BestMatch{d1,d2}(i,isNotNan));
            end
        end
    end
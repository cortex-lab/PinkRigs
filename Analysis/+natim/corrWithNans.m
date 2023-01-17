function C = corrWithNans(X1,X2)

    %%% Performs matrix correlation (row by row) when matrices contain nans

    if ~any(isnan(X1(:))) && ~any(isnan(X2(:)))
        C = corr(X1,X2); % still faster
    else
        % Will be slower 
        X = repmat(X1,[1 1 size(X2,2)]);
        Y = repmat(X2,[1 1 size(X1,2)]);
        Y = permute(Y, [1 3 2]);

        % Put nans in same place
        nanIdx = isnan(Y) | isnan(X);
        X(nanIdx) = nan;
        Y(nanIdx) = nan;

        % De-mean Columns:
        X = bsxfun(@minus,X,nansum(X,1)./sum(~isnan(X),1));
        Y = bsxfun(@minus,Y,nansum(Y,1)./sum(~isnan(Y),1));

        % Normalize by the L2-norm (Euclidean) of Rows:
        X = X.*repmat(sqrt(1./max(eps,nansum(abs(X).^2,1))),[size(X,1),1]);
        Y = Y.*repmat(sqrt(1./max(eps,nansum(abs(Y).^2,1))),[size(Y,1),1]);

        % Compute Pair-wise Correlation Coefficients:
        C = squeeze(nansum(X.*Y));
    end
end
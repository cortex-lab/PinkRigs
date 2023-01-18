function mouseBotRow(varargin)
    C = get (gca, 'CurrentPoint');
    title(gca, ['botRow = ',num2str(ceil(C(1,2)/15))]);
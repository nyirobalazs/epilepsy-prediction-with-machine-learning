function [signalsOut, labelsOut] = segSignals(signalsIn,labelsIn,targetLength,channels)

signalsOut = {};
labelsOut = {};

for idx_ch = channels
    for idx = 1:numel(signalsIn)

        x = signalsIn{idx};
        y = labelsIn(idx);

        % Oszlop vektorrá alakítás
        x = x(idx_ch,:)';

        % Számítsa ki a jelben található targetLength-mintadarabok számát
        numSigs = floor(length(x)/targetLength);

        if numSigs == 0
            continue;
        end

        x = x(1:numSigs*targetLength);

        % Hozzon létre egy mátrixot annyi oszlopból, ahány targetLength jel
        % van
        M = reshape(x,targetLength,numSigs); 

        % Ismételje meg a numSigs címkét
        y = repmat(y,[numSigs,1]);

        % Függőlegesen összefűzve cell array-okká
        signalsOut = [signalsOut; mat2cell(M.',ones(numSigs,1))]; 
        labelsOut = [labelsOut; cellstr(y)]; 
    end
end
labelsOut = categorical(labelsOut);

end
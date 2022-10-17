function [signalsOut, labelsOut] = segSignals(signalsIn,labelsIn,targetLength,channels)

signalsOut = {};
labelsOut = {};

for idx_ch = channels
    for idx = 1:numel(signalsIn)

        x = signalsIn{idx};
        y = labelsIn(idx);

        % Column to vector conversion
        x = x(idx_ch,:)';

        % Calculate the number of targetLength samples in the signal
        numSigs = floor(length(x)/targetLength);

        if numSigs == 0
            continue;
        end

        x = x(1:numSigs*targetLength);

        % Create a matrix of as many columns as there are targetLength signals
        M = reshape(x,targetLength,numSigs); 

        % Repeat the numSigs tag
        y = repmat(y,[numSigs,1]);

        % Vertically spliced into cell arrays
        signalsOut = [signalsOut; mat2cell(M.',ones(numSigs,1))]; 
        labelsOut = [labelsOut; cellstr(y)]; 
    end
end
labelsOut = categorical(labelsOut);

end

% nulla értékre állítja a rétegtömb rétegeinek összes paraméterének 
% tanulási sebességét.

function layers = freezeWeights(layers)

for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end

end




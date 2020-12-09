% findLayersToReplace(lgraph) megtalálja a réteggráf osztályozási rétegét 
% és az azt megelőző megtanulható (teljesen összefüggő vagy konvolúciós) 
% réteget.

function [learnableLayer,classLayer] = findLayersToReplace(lgraph)

if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Argumentumnak muszáj egy LayerGraph object-nek lennie.')
end

% Forrás-, cél- és rétegnevek beolvasása.
src = string(lgraph.Connections.Source);
dst = string(lgraph.Connections.Destination);
layerNames = string({lgraph.Layers.Name}');

% Klasszifikációs réteg megkeresése
isClassificationLayer = arrayfun(@(l) ...
    (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|isa(l,'nnet.layer.ClassificationLayer')), ...
    lgraph.Layers);

if sum(isClassificationLayer) ~= 1
    error('A Layer Graph-nak muszáj rendelkeznie egy single classification layer-el.')
end
classLayer = lgraph.Layers(isClassificationLayer);


% A rétegdiagramon az osztályozási rétegtől kezdve fordítva haladjon.
% Ha a hálózat elágazik, írjon ki hibát.
currentLayerIdx = find(isClassificationLayer);
while true
    
    if numel(currentLayerIdx) ~= 1
        error(' Nem megfelelő Learnable Layer')
    end
    
    currentLayerType = class(lgraph.Layers(currentLayerIdx));
    isLearnableLayer = ismember(currentLayerType, ...
        ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);
    
    if isLearnableLayer
        learnableLayer =  lgraph.Layers(currentLayerIdx);
        return
    end
    
    currentDstIdx = find(layerNames(currentLayerIdx) == dst);
    currentLayerIdx = find(src(currentDstIdx) == layerNames);
    
end

end


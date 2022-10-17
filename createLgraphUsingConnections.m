% lgraph = createLgraphUsingConnections(layers,connections) creates a layer graph, 
% in which the layers of the layer array |layers| are connected |connections|

function lgraph = createLgraphUsingConnections(layers,connections)

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end

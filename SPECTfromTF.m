function SPECTfromTF(dataIn,labelsIn,parentFolder,childFolder)
   
    % Célkönyvtár hivatkozás létrehozása
    imageRoot = fullfile(parentFolder,childFolder);

    % Belső változók létrehozása
    data = dataIn;
    labels = labelsIn;
    
    % szegmens hossz
    [~,signalLength] = size(data{1});

    % filterbant 
    fb = cwtfilterbank('SignalLength',signalLength,'VoicesPerOctave',12);
    r = size(data,1);
    
    for ii = 1:r
        fprintf("Create spectogram from TF %d of %d...\n", ii, r)
        cfs = abs(fb.wt(data{ii}));
        % indexelt kép RGB képpé alakítása
        im = ind2rgb(im2uint8(rescale(cfs)),jet(280));
        % kép mentési mappájának címe
        imgLoc = fullfile(imageRoot,char(labels(ii)));
        % kép neve és filekiterjeszése
        imFileName = strcat(char(labels(ii)),'_',num2str(ii),'.jpg');
        % ha a kért mappa nem létezik akkor létrehoz egyet
        if ~exist(imgLoc , 'dir')
            mkdir(imgLoc);
        end
        % Majd menti a képet
        imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
    end
end
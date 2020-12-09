function helperCreateRGBfromTF(EEGData,parentFolder,childFolder)

    imageRoot = fullfile(parentFolder,childFolder);

    data = EEGData.Data;
    labels = EEGData.Labels;

    [~,signalLength] = size(data);

    fb = cwtfilterbank('SignalLength',signalLength,'VoicesPerOctave',12);
    r = size(data,1);

    for ii = 1:r
        cfs = abs(fb.wt(data(ii,:)));
        im = ind2rgb(im2uint8(rescale(cfs)),jet(128));

        imgLoc = fullfile(imageRoot,char(labels(ii)));
        imFileName = strcat(char(labels(ii)),'_',num2str(ii),'.jpg');
        imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
    end
end
function SPECTfromTF(dataIn,labelsIn,parentFolder,childFolder)
   
    % Create target directory reference
    imageRoot = fullfile(parentFolder,childFolder);

    % Create internal variables
    data = dataIn;
    labels = labelsIn;
    
    % segment length
    [~,signalLength] = size(data{1});

    % filterbant 
    fb = cwtfilterbank('SignalLength',signalLength,'VoicesPerOctave',12);
    r = size(data,1);
    
    for ii = 1:r
        fprintf("Create spectogram from TF %d of %d...\n", ii, r)
        cfs = abs(fb.wt(data{ii}));
        % convert indexed image to RGB image
        im = ind2rgb(im2uint8(rescale(cfs)),jet(280));
        % title of the image save folder
        imgLoc = fullfile(imageRoot,char(labels(ii)));
        % image name and file extension
        imFileName = strcat(char(labels(ii)),'_',num2str(ii),'.jpg');
        % if the requested folder does not exist it will create one
        if ~exist(imgLoc , 'dir')
            mkdir(imgLoc);
        end
        % Then save the image
        imwrite(imresize(im,[224 224]),fullfile(imgLoc,imFileName));
    end
end

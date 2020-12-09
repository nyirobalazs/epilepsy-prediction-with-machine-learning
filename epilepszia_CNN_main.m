%%%
% A program fuuttatásához szükség van:
% - Deep Learning Toolbox Model for GoogLeNet Network
% - Wavelet Toolbox
% - Deep Learning toolbox
% ...kiegészítőkre
% Továbbá hogy az összes projekt file egy könyvtárban legyen és utánna
% nyissuk meg a main könyvtárat
%%%
%% Adatok betöltése

%betöltjük az egerekről készült EEG felvételeket
load dbNew.mat

%Összes EEG trial egy 3D-s (csatorna,idő,trial) mátrixba összesítése
for i=1:102
    EEG(:,:,i) = db{1,i}.eeg;
end

%% Time-Frequency Reprezentáció készítése
% tesztelési célra, hogy lássuk korrekt-e az elképzelés, egy-egy általunk
% választott normál(~n),roham(ictal~i) és roham utáni(postictal~p)
% szakaszon

Fs = 1000;
fb = cwtfilterbank('SignalLength',10000,...
    'SamplingFrequency',Fs,...
    'VoicesPerOctave',12);
sig_n = EEG(9,1:10000,1); 
sig_i = EEG(9,50001:60000,1);
sig_p = EEG(9,140001:150000,1);
[cfs_n,frq_n] = wt(fb,sig_n); [cfs_i,frq_i] = wt(fb,sig_i); [cfs_p,frq_p] = wt(fb,sig_p);
t = (0:9999)/Fs;

figure;
subplot(3,1,1)
pcolor(t,frq_n,abs(cfs_n))
set(gca,'yscale','log');shading interp;axis tight;
title('Normál');xlabel('Idő (sec)');ylabel('Frekvencia (Hz)')

subplot(3,1,2)
pcolor(t,frq_i,abs(cfs_i))
set(gca,'yscale','log');shading interp;axis tight;
title('Ictal');xlabel('Idő (sec)');ylabel('Frekvencia (Hz)')

subplot(3,1,3)
pcolor(t,frq_p,abs(cfs_p))
set(gca,'yscale','log');shading interp;axis tight;
title('Post-ictal');xlabel('Idő (sec)');ylabel('Frekvencia (Hz)')


%% Adatok előkészítése

%Adatok dimenzióinak kinyerése
[numb_chen,pnts,trials]=size(EEG);

%EEG roham minták kiválasztott csatornáinak(channels) feldarabolása normál,
%ictal és postictal szakaszokra. Szétválasztása training és validation
%adatokra(90-10%-os arányban),majd tovább osztása a targetLength-el
%megyegyező méretűre
channels = [2 6 7 9 11];
targetLength = 5000; %leghatékonyabb az 5000-es méret
[SignalsT,LabelsT,SignalsV,LabelsV]=prepOwnData(db,EEG,targetLength,channels);

% parentFolder-t át kell írni egy egy valid mappa elérési címre
% 1.a - Training data képpé alakítása
parentFolder = 'C:\Users\Nyírő Balázs\OneDrive - Kormányzati Informatikai Fejlesztési Ügynökség\Egyetem\Önlab\teszt data\v_2';
set_label = '11_10';
set_name = '_Train';
childFolder = append(set_label,set_name);
%spektogram reprezentáció morlet wavelet segítségével
SPECTfromTF(SignalsT,LabelsT,parentFolder,childFolder); 

% 1.b - Beolvas datastore-ba a könyebb kezelés érdekében
imgsTrain = imageDatastore(fullfile(parentFolder,childFolder),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
clc

% ugyan ezt a validációs adatokkal is megcsináljuk
%2.a - Validation data képpé alakítása
set_name = '_Valid';
childFolder = append(set_label,set_name);
SPECTfromTF(SignalsV,LabelsV,parentFolder,childFolder);

%2.b - Datastore
imgsValidation = imageDatastore(fullfile(parentFolder,childFolder),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
clc

%% GoogLeNet

% GoogLeNet előre tanított háló betöltése 
net = googlenet;

% GoogLeNet Network paramétereinek megváltoztatása
% 1. Legutolsó droput layer helyettesítése egy 0.6-os értékű dropout layerrel
% New dropout
% Layer Graph kinyerése. Ha a neurális háló egy SeriesNetwork object,
% mint pl AlexNet, VGG-16 vagy a VGG-19, akkor konvertálja a list of layers-t
% net.Layers-t egy layer graph.
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

newDropoutLayer = dropoutLayer(0.6,'Name','new_Dropout');
lgraph = replaceLayer(lgraph,'pool5-drop_7x7_s1',newDropoutLayer);


% 2. Fully connected layer cserélése egy olyanra ahol 3 class van (ahány kategóriánk a labelekben)
numClasses = numel(categories(imgsTrain.Labels));

[learnableLayer,classLayer] = findLayersToReplace(lgraph);

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newConnectedLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',5, ...
        'BiasLearnRateFactor',5);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newConnectedLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',5, ...
        'BiasLearnRateFactor',5);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newConnectedLayer);


% 3. Az eredeti classification layer kicserélése egy olyanra amiben nincsenek
%    label-ek. Ezeket majd a betanítás során autómatikusan beállítja magának.
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% Háló struktúra megtekintés (pcionális)
% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

% Freeze(azokat a rétegeket melyek az alapvető struktúrák feliserését végezik
% befagyasztjuk, azaz ezeknek az értékeit nem változtatja majd a betanulás során) 
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

% Training Options beállítása az általunk kívánt értékekre
options = trainingOptions('sgdm',...
    'MiniBatchSize',20,...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4,...
    'ValidationData',imgsValidation,... 
    'ValidationFrequency',10,...
    'Verbose',1,...
    'ExecutionEnvironment','gpu',...
    'Plots','training-progress');
rng default

%% Háló betanítása
% training futás elindítása
trainedGN = trainNetwork(imgsTrain,lgraph,options);

%% Pontosság kiszámolása
% pontosság számolása validációs adatokra
trainedGN.Layers(end)

[YPred,probs] = classify(trainedGN,imgsValidation);
accuracy = mean(YPred==imgsValidation.Labels);
disp(['GoogLeNet Accuracy: ',num2str(100*accuracy),'%'])

%% Klasszifikálás - teszt

% validation képekre --> random válasz 4-et a validációs képek közül
% majd ezeket klasszifikálja. Ha ezt helyesen teszi az adott szakasz
% spektogramja felett zöld felirat jelzi hogy melyik szakasz lett
% megállapítva és mekkora pontossággal. Ha nem találta el a felirat piros
% illetve megjelenik hogy mi a valódi besorolása az adott szakasznak

% a klasszifikálás hosszát mérjük
tic
[YPred,probs] = classify(trainedGN,imgsValidation);
toc

% plot
idx = randperm(numel(imgsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imgsValidation,idx(i));
    imshow(I)
    if YPred(idx(i))=='n'
        label='Normal';
    elseif YPred(idx(i))=='i'
        label='Ictal';
    elseif YPred(idx(i))=='p'
        label='Post-ictal';
    end
        
    if YPred(idx(i))== imgsValidation.Labels(idx(i))
        title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%",'Color','#77AC30');
    else
        title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%" + " --> " + char(imgsValidation.Labels(idx(i))) ,'Color','red');
    end
end

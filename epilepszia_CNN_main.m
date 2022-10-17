%%%
% Required to run the program:
% - Deep Learning Toolbox Model for GoogLeNet Network
% - Wavelet Toolbox
% - Deep Learning toolbox
% ... accessories:
% Furthermore, all project files must be in one directory and then open the main directory
%%%

%% Load data
%load EEG recordings from mice
load dbNew.mat

%Aggregation of all EEG trials into a 3D (channel,time,trial) matrix
for i=1:102
    EEG(:,:,i) = db{1,i}.eeg;
end

%% Create Time-Frequency Representation
% for testing purposes, to see if the idea is correct, for one of our
% normal(~n), seizure(ictal~i) and post-seizure(postictal~p) phase

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


%% Pre-processing of data

%Data dimensions extraction
[numb_chen,pnts,trials]=size(EEG);

%EEG seizure samples of selected channels are split into normal channels,
%ictal and postictal sections. Separation training and validation
%data (90-10% ratio) and further division by targetLength into a uniform size
channels = [2 6 7 9 11];
targetLength = 5000; %leghatékonyabb az 5000-es méret
[SignalsT,LabelsT,SignalsV,LabelsV]=prepOwnData(db,EEG,targetLength,channels);

% parentFolder must be rewritten to a valid folder path
% 1.a - Convert Training data to image
parentFolder = 'C:\Users\Nyírő Balázs\OneDrive - Kormányzati Informatikai Fejlesztési Ügynökség\Egyetem\Önlab\teszt data\v_2'; %modify this folder path
set_label = '11_10';
set_name = '_Train';
childFolder = append(set_label,set_name);
% spectrogram representation using morlet wavelet
SPECTfromTF(SignalsT,LabelsT,parentFolder,childFolder); 

% 1.b - Scan to datastore for easier management
imgsTrain = imageDatastore(fullfile(parentFolder,childFolder),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
clc

% we do the same with the validation data
%2.a - Convert Validation data to image
set_name = '_Valid';
childFolder = append(set_label,set_name);
SPECTfromTF(SignalsV,LabelsV,parentFolder,childFolder);

%2.b - Datastore
imgsValidation = imageDatastore(fullfile(parentFolder,childFolder),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
clc

%% GoogLeNet

% Load GoogLeNet pre-trained mesh 
net = googlenet;

% Change GoogLeNet Network parameters
% 1. Replace the last dropout layer with a dropout layer of 0.6
% New dropout
% Retrieve Layer Graph. If the neural network is a SeriesNetwork object,
% such as AlexNet, VGG-16 or VGG-19, then convert the list of layers
% net.Layers to a layer graph.
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

newDropoutLayer = dropoutLayer(0.6,'Name','new_Dropout');
lgraph = replaceLayer(lgraph,'pool5-drop_7x7_s1',newDropoutLayer);


% 2. Replace Fully connected layer with a layer with 3 classes (as many categories in the labels)
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


% 3. Replace the original classification layer with one that does not
% contain labels. These will be set automatically during training.
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% View of mesh structure(optional)
% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

% Freeze(the layers that are used to lift the basic structures
% will be frozen, i.e. their values will not change during the learning process) 
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

% Set Training Options to the values you want
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

%% Training
% Start training run
trainedGN = trainNetwork(imgsTrain,lgraph,options);

%% Calculation of accuracy
% Calculation of accuracy for validation data
trainedGN.Layers(end)

[YPred,probs] = classify(trainedGN,imgsValidation);
accuracy = mean(YPred==imgsValidation.Labels);
disp(['GoogLeNet Accuracy: ',num2str(100*accuracy),'%'])

%% Classification - test

% validation images --> random answer 4 of the validation images
% then classifies them. If this is done correctly the given section
% spectrogram of a given section, a green label above the 
% spectrogram indicates which section has been
% identified and with what accuracy. If not found, the label is red
% and the actual classification of the given section is displayed

% the length of the classification is measured
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

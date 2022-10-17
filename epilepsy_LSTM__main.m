%%%
% Required to run the program:
% - Deep Learning Toolbox Model for GoogLeNet Network
% - Wavelet Toolbox
% - Deep Learning toolbox
% Furthermore, all project files must be in one directory and then
% open the main directory
%%%
%% Load data

%load the EEG recordings of the mice
load dbNew.mat

%EEG trial aggregation into a 3D (channel,time,trial) matrix
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


%% Data preparation

%Data dimensions extraction
[numb_chen,pnts,trials]=size(EEG);

%EEG seizure samples of selected channels are split into normal channels,
%ictal and post ictal sections. Separation training and validation
%data (90-10% ratio) and further division by targetLength into a uniform size
channels = [2 6 7 9 11];
targetLength = 5000; %leghatékonyabb az 5000-es méret
[SignalsT,LabelsT,SignalsV,LabelsV]=prepOwnData(db,EEG,targetLength,channels);

% Instantaneous frequency calculation
fs = 1000;
instfreqTrain = cellfun(@(x)instfreq(x,fs)',SignalsT,'UniformOutput',false);
instfreqTest = cellfun(@(x)instfreq(x,fs)',SignalsV,'UniformOutput',false);

% Spectral entropy calculation
pentropyTrain = cellfun(@(x)pentropy(x,fs)',SignalsT,'UniformOutput',false);
pentropyTest = cellfun(@(x)pentropy(x,fs)',SignalsV,'UniformOutput',false);

TrainS_2 = cellfun(@(x,y)[x;y],instfreqTrain,pentropyTrain,'UniformOutput',false);
TestS_2 = cellfun(@(x,y)[x;y],instfreqTest,pentropyTest,'UniformOutput',false);

%Normalisation
XV = [TrainS_2{:}];
mu = mean(XV,2);
sg = std(XV,[],2);

XTrainSD = TrainS_2;
XTrainSD = cellfun(@(x)(x-mu)./sg,XTrainSD,'UniformOutput',false);

XTestSD = TestS_2;
XTestSD = cellfun(@(x)(x-mu)./sg,XTestSD,'UniformOutput',false);

%% Defining LSTM Network architecture

layers = [ ...
    sequenceInputLayer(2)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
    ]

options = trainingOptions('adam', ...  % sztochasztikus gradiens -> sgdm 
    'MaxEpochs',45, ...            
    'MiniBatchSize', 90, ...       
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

%% Train network
% training run start
net = trainNetwork(XTrainSD,LabelsT,layers,options);

%% Accuracy test
% accuracy calculation 
trainPred = classify(net,XTrainSD);
LSTMAccuracy = sum(trainPred == LabelsT)/numel(LabelsT)*100

%plot of convolution matrix
figure
confusionchart(LabelsT,trainPred,'ColumnSummary','Normalizált oszlopok',...
              'RowSummary','Normalizált sorok','Title','LSTM konvolúciós mátrix');

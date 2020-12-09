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
%ictal és post ictal szakaszokra. Szétválasztása training és validation
%adatokra(90-10%-os arányban),majd tovább osztása a targetLength-el
%megyegyező méretűre
channels = [2 6 7 9 11];
targetLength = 5000; %leghatékonyabb az 5000-es méret
[SignalsT,LabelsT,SignalsV,LabelsV]=prepOwnData(db,EEG,targetLength,channels);

% Pillanatnyi frekvencia számítása
fs = 1000;
instfreqTrain = cellfun(@(x)instfreq(x,fs)',SignalsT,'UniformOutput',false);
instfreqTest = cellfun(@(x)instfreq(x,fs)',SignalsV,'UniformOutput',false);

% Spektrális entrópia számítása
pentropyTrain = cellfun(@(x)pentropy(x,fs)',SignalsT,'UniformOutput',false);
pentropyTest = cellfun(@(x)pentropy(x,fs)',SignalsV,'UniformOutput',false);

TrainS_2 = cellfun(@(x,y)[x;y],instfreqTrain,pentropyTrain,'UniformOutput',false);
TestS_2 = cellfun(@(x,y)[x;y],instfreqTest,pentropyTest,'UniformOutput',false);

%Normalizálás
XV = [TrainS_2{:}];
mu = mean(XV,2);
sg = std(XV,[],2);

XTrainSD = TrainS_2;
XTrainSD = cellfun(@(x)(x-mu)./sg,XTrainSD,'UniformOutput',false);

XTestSD = TestS_2;
XTestSD = cellfun(@(x)(x-mu)./sg,XTestSD,'UniformOutput',false);

%%  LSTM Network architektúra definiálása

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
% training futás elindítása
net = trainNetwork(XTrainSD,LabelsT,layers,options);

%% Accuracy test
% pontosság számolása 
trainPred = classify(net,XTrainSD);
LSTMAccuracy = sum(trainPred == LabelsT)/numel(LabelsT)*100

%konvolúciós mátrix ábrázolása
figure
confusionchart(LabelsT,trainPred,'ColumnSummary','Normalizált oszlopok',...
              'RowSummary','Normalizált sorok','Title','LSTM konvolúciós mátrix');
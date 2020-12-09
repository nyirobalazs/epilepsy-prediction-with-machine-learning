function [SignalsT,LabelsT,SignalsV,LabelsV]=prepOwnData(database,EEG,targetLength,channels)

db = database;
%% Adatok betöltése és szegmensek szétválasztása
%Összes EEG trial egy 3D-s (csatorna,idő,trial) mátrixba összesítése
%db = database;
[numb_chen,pnts,trials]=size(EEG);

% Trialok normál,ictal és post-ictal részekre osztása és címkék létrehozása
j=1;
for i=1:trials-2
    EEG_ML{j} = EEG(:,1:49999,i); Labels{j} = 'n';
    EEG_ML{j+1} = EEG(:,50000:50000+(db{1,i}.seizureEnd-db{1,i}.stimStart)*1000,i); Labels{j+1} = 'i';
    EEG_ML{j+2} = EEG(:,50000+(db{1,i}.seizureEnd-db{1,i}.stimStart)*1000:end,i); Labels{j+2} = 'p';
    j = j+3;
end
clc
EEG_ML = EEG_ML'; Labels = Labels'; %invertálás oszlopba

%% Adatok szétválasztása betanítási és validálási részre(90%:10%-os arányban)
%Random sorrend létrehozása
numEEGs = numel(EEG_ML);
idx = randperm(numEEGs);
N = floor(0.9 * numEEGs);

%Betanítási adatok
idxTrain = idx(1:N);
SignalsTrain = EEG_ML(idxTrain);
LabelsTrain = Labels(idxTrain);

%Validálási adatok
idxValidation = idx(N+1:end);
SignalsValidation = EEG_ML(idxValidation);
LabelsValidation = Labels(idxValidation);

%% Tovább szakaszolás
% A segmentSignals funkció törli azokat a szakaszokat amiket a
% targetLength-nél rövidebbek és több egységnyi hosszú (targetLength hosszúságú) részre 
% vágja azokat amik hoszabbak.

[SignalsT, LabelsT] = segSignals(SignalsTrain,LabelsTrain,targetLength,channels);
[SignalsV, LabelsV] = segSignals(SignalsValidation,LabelsValidation,targetLength,channels);

for z=1:2
    if z==1
        Signals = SignalsT;
        Labels = LabelsT;
        name = "Betanítási adatok";
    elseif z==2
        Signals = SignalsV;
        Labels = LabelsV;
        name = "Validálási adatok";
    end
    
    fprintf("%s arányai rendezés előtt\n",name)
    summary(Labels)
    % Osszuk őket 3 felé
    normS = Signals(Labels=='n');
    normL = Labels(Labels=='n');

    ictS = Signals(Labels=='i');
    ictL = Labels(Labels=='i');

    postS = Signals(Labels=='p');
    postL = Labels(Labels=='p');
    
    % az adott szegmenseket azonos számura hozni
    summ = [numel(normL) numel(ictL) numel(postL)];
    
    if max(summ)-min(summ) <= min(summ)
        %ha elég kicsi a különbség akkor minden vektort a leghoszabb 
        %vektor méreteire bővítünk úgy hogy az elejéből akkora szeletet
        %csípünk le és fűzünk a végére amekkora a különbség
        normS = [normS;normS(1:max(summ)-numel(normL))];
        normL = [normL;normL(1:max(summ)-numel(normL))];

        ictS = [ictS;ictS(1:max(summ)-numel(ictL))];
        ictL = [ictL;ictL(1:max(summ)-numel(ictL))];

        postS = [postS;postS(1:max(summ)-numel(postL))];
        postL = [postL;postL(1:max(summ)-numel(postL))];
        
    elseif max(summ)-min(summ) > min(summ)
        %ha túl nagy a mennyiségbeli különbség a vektorok között
        %akkor annyiadszor ismételjük meg egymás után a vektort
        %amekkora szoros különbség van közötte és a leghoszabb között
        dev = floor(max(summ)/numel(normL));
        normS = repmat(normS,dev,1);
        normL = repmat(normL,dev,1);
        
        dev = floor(max(summ)/numel(ictL));
        ictS = repmat(ictS,dev,1);
        ictL = repmat(ictL,dev,1);
        
        dev = floor(max(summ)/numel(postL));
        postS = repmat(postS,dev,1); 
        postL = repmat(postL,dev,1);
        
        %Aztán a maradékot kiegészítjük a már korábban használt módon
        normS = [normS;normS(1:max(summ)-numel(normL))];
        normL = [normL;normL(1:max(summ)-numel(normL))];

        ictS = [ictS;ictS(1:max(summ)-numel(ictL))];
        ictL = [ictL;ictL(1:max(summ)-numel(ictL))];

        postS = [postS;postS(1:max(summ)-numel(postL))];
        postL = [postL;postL(1:max(summ)-numel(postL))];
    end
    
    if z==1
        SignalsT = [normS;ictS;postS];
        LabelsT = [normL;ictL;postL];
        % ellenőrzés
        fprintf("%s arányai rendezés után\n",name)
        summary(LabelsT)
    elseif z==2
        SignalsV = [normS;ictS;postS];
        LabelsV = [normL;ictL;postL];
        % ellenőrzés
        fprintf("%s arányai rendezés után\n",name)
        summary(LabelsV)
    end
end

end
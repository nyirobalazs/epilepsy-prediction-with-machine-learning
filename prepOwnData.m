function [SignalsT,LabelsT,SignalsV,LabelsV]=prepOwnData(database,EEG,targetLength,channels)

db = database;
%% Loading data and splitting segments
%All EEG trials aggregated into a 3D (channel,time,trial) matrix
%db = database;
[numb_chen,pnts,trials]=size(EEG);

% Divide trials into normal, ictal and post-ictal parts and create labels
j=1;
for i=1:trials-2
    EEG_ML{j} = EEG(:,1:49999,i); Labels{j} = 'n';
    EEG_ML{j+1} = EEG(:,50000:50000+(db{1,i}.seizureEnd-db{1,i}.stimStart)*1000,i); Labels{j+1} = 'i';
    EEG_ML{j+2} = EEG(:,50000+(db{1,i}.seizureEnd-db{1,i}.stimStart)*1000:end,i); Labels{j+2} = 'p';
    j = j+3;
end
clc
EEG_ML = EEG_ML'; Labels = Labels'; %invert to column

%% Separation of data into training and validation (90%:10%)
%Create random order
numEEGs = numel(EEG_ML);
idx = randperm(numEEGs);
N = floor(0.9 * numEEGs);

%Training data
idxTrain = idx(1:N);
SignalsTrain = EEG_ML(idxTrain);
LabelsTrain = Labels(idxTrain);

%Validation data
idxValidation = idx(N+1:end);
SignalsValidation = EEG_ML(idxValidation);
LabelsValidation = Labels(idxValidation);

%% Segmenting further
% The segmentSignals function deletes segments that have been
% shorter than targetLength and into segments of several units length (targetLength). 
% cuts those that are longer.

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
    % Divide them into 3 halves
    normS = Signals(Labels=='n');
    normL = Labels(Labels=='n');

    ictS = Signals(Labels=='i');
    ictL = Labels(Labels=='i');

    postS = Signals(Labels=='p');
    postL = Labels(Labels=='p');
    
    % to bring the given segments to the same number
    summ = [numel(normL) numel(ictL) numel(postL)];
    
    if max(summ)-min(summ) <= min(summ)
        %if the difference is small enough then all vectors are the most 
        %vectors so that a slice from the beginning of the vector
        %clipped and appended to the end as large as the difference
        normS = [normS;normS(1:max(summ)-numel(normL))];
        normL = [normL;normL(1:max(summ)-numel(normL))];

        ictS = [ictS;ictS(1:max(summ)-numel(ictL))];
        ictL = [ictL;ictL(1:max(summ)-numel(ictL))];

        postS = [postS;postS(1:max(summ)-numel(postL))];
        postL = [postL;postL(1:max(summ)-numel(postL))];
        
    elseif max(summ)-min(summ) > min(summ)
        %if the quantitative difference between the vectors is too large
        %then repeat the vector as many times in a row
        %the closest difference between it and the largest
        dev = floor(max(summ)/numel(normL));
        normS = repmat(normS,dev,1);
        normL = repmat(normL,dev,1);
        
        dev = floor(max(summ)/numel(ictL));
        ictS = repmat(ictS,dev,1);
        ictL = repmat(ictL,dev,1);
        
        dev = floor(max(summ)/numel(postL));
        postS = repmat(postS,dev,1); 
        postL = repmat(postL,dev,1);
        
        %Then the remainder is added in the same way as before
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

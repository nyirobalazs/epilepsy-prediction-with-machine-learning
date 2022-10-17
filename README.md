# **Epilepsy detection artificial with machine learning**

The aim of my bachelor thesis was to classify the three phases of barking seizures in mice using a convolutional neural network(CNN) and bidirectional LSTM (BiLSTM). The project has demonstrated that it is possible to quickly and accurately predict the occurrence of an epileptic seizure up to 30 minutes before the seizure, after proper wavelet decomposition for CNN and based on the calculated Instantaneous Frequency and Spectral Entropy values for BiLSTM.
<br>
![thesis](https://github.com/nyirobalazs/epilepsy-prediction-with-machine-learning/blob/main/assets/Untitled%20(4).png)
<br>

## Background

Epilepsy is a disorder affecting nearly 50 million people worldwide, of whom 60-80 thousand live in our country. It is a disorder of the brain in which neurons fire synchronously, either locally or globally, when released from inhibition, triggering a seizure. This symptom not only affects their family life, relationships and work, but if left untreated, it can also threaten them daily with severe injuries - suffered during seizures - that can result in an epileptic patient being 2-3 times more likely to die prematurely than their non-epileptic peers. To make matters worse, 50-75,000 of these people die from a specific form of epilepsy - SUDEP (sudden death during a seizure) - which could be significantly reduced by daily monitoring and seizure recognition with appropriate support.


## Project

In my thesis, my goal was to create a machine learning-based model in a MATLAB environment that can detect as accurately as possible the brain electrical activity derived from the brain surface or from an electroencephalograph (EEG) placed outside the head. It has already been outlined in the literature search that the best fitting algorithms for the purpose will be convolutional neural networks (CNNs) and long short-term memories (LSTMs). Their success lies in the fact that while the former is better able to learn the patterns of signals recorded from epileptic patients, the latter is better able to recognise the temporal correlations of the patterns. Taking previous research further, my work has improved the prediction accuracy of the architectures by using a transfer learning model in the case of convolutional meshes and by calculating the instantaneous frequency and spectral entropy of the signals in the case of LSTM. Furthermore, the above-mentioned architectures were run with different epoch window sizes (2000-10,000 milliseconds) and training values , and optimization functions to find the most optimal parameters. My results showed that the highest validation accuracy on the given database can be achieved with an epoch window of 5000 milliseconds without overlapping and with Adam optimization. Thanks to these improvements, I was able to achieve a validation accuracy of 92.71% for CNN and 93.49% for LSTM. 

### The preprocessing of preictal,ictal and post ictal segments

![CNN preprocessing results](https://github.com/nyirobalazs/epilepsy-prediction-with-machine-learning/blob/main/assets/raw_data_wavelet(11-08)_1.jpg)
<br>

### Results of classification with the trained CNN

![CNN predict](https://github.com/nyirobalazs/epilepsy-prediction-with-machine-learning/blob/main/assets/CNN_train02(12.10)_test_plot02.jpg)
<br>

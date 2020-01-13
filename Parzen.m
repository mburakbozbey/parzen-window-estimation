clear all;
clc;

n_ = 86;  % Please change n to reproduce results for n = 86
h_ = .275;  % Please change h to reproduce results for h = 0.275, 0.1

dataset = readmatrix('path/to/dataset/');

% Variables and placeholders are defined.

pc = 0.75 ;     % percentage 75%
rng('default')  % For reproducibility
classNum = 8;

dim = 100;
means = zeros(classNum, dim, 6);
C = zeros(dim, dim, classNum, 6);
priors = zeros(classNum, 6);

D = [];
T = [];

% Split each class, %75 to D, %25 to T
for k=1:8
    idx = dataset(:,1) == k;
    sampleC = dataset(idx,:);
    train = sampleC(1:round(pc*size(sampleC,1)),:);  % Split D set for kth class
    test = sampleC(round(pc*size(sampleC,1)):end,:); % Split T set for kth class
    D = [D;train];
    T = [T;test];
end

% Split patterns and classes of D & T set
dataClass = D(:,1);
dataPatterns = D(:,[2:end-1]);
dataRowCount = size(dataPatterns, 1);

% Parzen
testClass = T(:,1);
testPatterns = T(:,[2:end,]);
testRowCount = size(testPatterns, 1);

% Calculate prior probability for each class

for k=1:8
    for m=1:6
        numDataset = numel(D(:,[2:end]));
        idx = D(:,1) == k;
        sampleC = D(idx,1+(m-1)*dim:m*dim+1);
        samplePatterns = sampleC(:,[2:end]);
        priors(k, m) = length(samplePatterns)/length(D);           % Priors
    end
end

predScore = [];
predClass = [];

% Apply simplified bayes rule to T set with pdf values of 
% Parzen Windows Estimation
sumPDF = 0;
for j=1:testRowCount
    P = zeros(8, 6);
    for m=1:6
        x = testPatterns(j, 1+(m-1)*dim:dim*m);
        pdf = zeros(8,6);
        for c=1:8
            idx = D(:,1) == c;
            sampleC =  D(idx,1+(m-1)*dim:m*dim+1);
            xi = sampleC(1:sum(idx),[2:end]);
            pdf = sum(mvksdensity(x, xi, 'Bandwidth', h_));
            P(c, m) =  priors(c,m)*log(pdf + 0.00001);
            sumPDF = sumPDF + pdf;
        end
    end
    [score idx] = max(sum(P'));
    predScore = [predScore;score];
    predClass = [predClass;idx];
end


% Create confusion matrix and calculate precision & recall for T set
accuracy = 1 - length(find(testClass~=predClass))/testRowCount; 
disp("Accuracy:");
disp(accuracy);
confM = confusionmat(testClass,predClass);
confM = confM';
confM = confM + eps; % Addition by epsilon due to division by zero error, etc.
testPrecision = diag(confM)./sum(confM,2);
testRecall = diag(confM)./sum(confM,1)';

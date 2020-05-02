% waveSpeedCalcGit wrapper for test data generation
%Process wave speed
clear; clc; close all;

%% Load file(s)
baseDir = cd;
cd('G:\My Drive\Dylan\Classes\2020 - Spring\ME 759 - HPC for Applications in Engineering\FinalProject\new_code\');
[f,d] = uigetfile('*.lvm');
filenames.lvm = [d '\' f];
% filenames = [];
cd(baseDir);

%% Options
options.collectionMethod = 'accelerometer';
options.accDataYesNo = 1;
options.numAcc = 2;

options.accColumns = [3 4]; %A1 and A2
options.measOrder = [1 2];
options.signCorrection = [1 1];

options.waveSpeedMethod = 'XCorr';
options.window = [0 1];
options.travelDist = 10; %travel distance in mm

options.tapperColumns = 2;
options.plotYesNo = 0;

options.filterBandWave = [150 5000];
options.deltaWSThresh = 100;
options.filterLowPass = 10;
options.nanFill = 1;

options.normxcorr2_mex = 0;

%% Process data
tic
data = waveSpeedCalcGit(filenames,options);
toc

%% Save acc data for single tap
% compute offset to first wave speed measurement
% dataToSave = data.processedData.wave.push{1,1};
% writematrix(dataToSave,'singleTap_firstCheck2.csv');

%% Plot comparison of Matlab and C wave speeds
push = csvread('processed_push.csv');
% release = csvread('processed_release.csv');
% tapInds = csvread('sortedTap.csv');

timeVect = linspace(0,60,3000);

figure
plot(timeVect,push,'b');
hold on;
% plot(timeVect(1:end-1),release,'b--');
plot(timeVect,data.processedData.waveSpeed.unfilt.push{1,1},'r');
% plot(timeVect(1:end-1),data.processedData.waveSpeed.unfilt.release{1,1},'r--');
% legend('C, push','C, release','Matlab, push','Matlab, release');
legend('C, push','Matlab, push');
xlabel('Time [s]');
ylabel('Wave Speed [m/s]');

return
%% Load extra data
clear filteredAcc1 filteredAcc2 pushPullIndices rvals pushIndices...
    pullIndices zpush zpull zlead ztrail rvals_parsed
filteredAcc1 = csvread('filtered_acc1.csv');
filteredAcc2 = csvread('filtered_acc2.csv');
pushPullIndices = csvread('push_pull_indices.csv');
rvals = csvread('r_vals.csv');

pushIndices = pushPullIndices(1:3001);
pullIndices = pushPullIndices(3002:end);

zpush = zeros(1,length(pushIndices));
zpull = zeros(1,length(pullIndices));

zlead = zeros(1,length(data.params.tapTiming.leading));
ztrail = zeros(1,length(data.params.tapTiming.trailing));

for ii = 1:length(pushIndices)-1
    rvals_parsed(:,ii) = rvals(((ii-1)*400)+1:(ii*400));
end

%% Extra plot(s)
close all;

figure
% plot(data.rawData.lvm.wave(:,1),'b');
hold on;
% plot(data.rawData.lvm.wave(:,2),'r');
plot(filteredAcc1,'b--');
plot(filteredAcc2,'r--');

plot(data.params.tapTiming.leading,zlead,'gx','markersize',7,'linewidth',1.5);
plot(data.params.tapTiming.trailing,ztrail,'go','markersize',7,'linewidth',1.5);
plot(pushIndices,zpush,'kx','markersize',5,'linewidth',1.5);
plot(pullIndices,zpull,'ko','markersize',5,'linewidth',1.5);

for ii = 1:size(rvals_parsed,2)
    plot(pushIndices(ii):1:pushIndices(ii)+399,rvals_parsed(:,ii),...
        'color',[1,0.5,0]);
end

for ii = 1:size(rvals_parsed,2)
    jack_rs = data.params.waveCorrelation.rVals(:,:,ii);
    jack_rs = jack_rs{1};
    ljack = length(jack_rs);
    plot(pushIndices(ii):1:pushIndices(ii)+ljack-1,jack_rs,...
        'color',[0.5,1,0]);
end


figure
plot(rvals_parsed);
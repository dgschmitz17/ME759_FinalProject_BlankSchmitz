function [output] = waveSpeedCalcGit(filenames,optionsIn,rawData)
% This function defines options for processing of wave speed data and
% ancillary data (e.g., forces, EMG, etc.), and calls functions to load
% files, calculate wave speeds, process wave speeds, process ancillary data,
% and plot results
% 
% Handles in vivo and ex vivo data; can utilize a number of wave speed
% calculation methods.
% 
% for accelerometer method, 'filenames' should be a structure with field,
% 'filenames.lvm', that is the full path to the labview collection file. It
% may also contain a field, 'filenames.mat', that has additional data
% (usually force data collected separately)
% 
% for ultrasound method, 'filenames' should contain the field,
% 'filenames.mat', which contains the processed (post-speckle-tracking)
% ultrasound data. It may also contain a field, 'filenames.lvm' that
% contains loading data.
%
% 'OptionsIn' is a structure including all relevant options. For more
% information, open 'waveSpeedOptions.m'. If 'OptionsIn' is left empty, or
% if relevant fields are left empty, default values will be used.
% 
% Created by Jack Martin - 05/25/17; additionally uses code written by Scott
% Brandon

% Define options
if exist('optionsIn','var')
    options = wsOptions(filenames,optionsIn);
else
    options = wsOptions();
end

% Editited below to allow using rawData structure rather than importing lvm
% files - Jack Martin - 11/12/18
if nargin < 3
    % Load relevant files
    if exist('filenames','var')
        [filenames,rawData] = wsFileImport(filenames,options);
    else
        [filenames,rawData] = wsFileImport([],options);
    end
else
    if isempty(rawData)
        % Load relevant files
        if exist('filenames','var')
            [filenames,rawData] = wsFileImport(filenames,options);
        else
            [filenames,rawData] = wsFileImport([],options);
        end
    end
end

% Determine tap indices
params.tapTiming = wsTapTiming(rawData,options);

% Determine window bounds for static windows
[params] = windowBounds(rawData,options,params);

% Prepare data for wave speed calculation
[processedData,params] = ...
    wsProcessRawData(rawData,options,params,filenames);

% Calculate wave speed
switch options.waveSpeedMethod
    case 'XCorr'
        [rawData.waveSpeed,params] = wsCalcXCorr(processedData,params,options);
    case 'P2P'
        [rawData.waveSpeed,params] = wsCalcP2P(processedData,params,options);
    case 'frequency'
        [rawData.waveSpeed, processedData.spectra] = wsCalcFreq(processedData,params,options);
end

% Edit/Clean wave speed data
processedData.waveSpeed = wsProcessWaveSpeed(rawData,params,options);

% Plot results
figHandles = wsPlot(rawData,processedData,params,options);

% Defining outputs
output.filenames = filenames;
output.options = options;
output.rawData = rawData;
output.params = params;
output.processedData = processedData;
if ~isempty(figHandles)
    output.figures = figHandles;
end


function [optionsOut] = wsOptions(filenames,optionsIn)
% This function defines all options not specified by the user.
%
% Jack Martin - 05/25/17

if exist('optionsIn','var')
    options = optionsIn;
else
    options = [];
end

if isempty(filenames)
    options.fileSelectionMode = if_dne(options,'fileSelectionMode','choose');
%     options.fileSelectionMode = if_dne(options,'fileSelectionMode','recent');
    options.defaultDirectory = if_dne(options,'defaultDirectory',pwd);
end

% Wave Data collection method
options.collectionMethod = if_dne(options,'collectionMethod','accelerometer');
% options.collectionMethod = if_dne(options,'collectionMethod','ultrasound');

% Method for calculating wave speed. 
options.waveSpeedMethod = if_dne(options,'waveSpeedMethod','XCorr');
% options.waveSpeedMethod = if_dne(options,'waveSpeedMethod','frequency');
% options.waveSpeedMethod = if_dne(options,'waveSpeedMethod','P2P');

% Was load data (e.g., joint torque) collected? (1=yes, 0=no)
options.loadDataYesNo = if_dne(options,'loadDataYesNo',0);
if options.loadDataYesNo
    % What type of file is the load data stored in?
    options.loadDataLoc = if_dne(options,'loadDataLoc','lvm');
%     options.loadDataLoc = if_dne(options,'loadDataLoc','mat');
end

% Was positional data (e.g., joint angle) collected? (1=yes, 0=no)
options.posDataYesNo = if_dne(options,'posDataYesNo',0);

% Was EMG data collected? (1=yes, 0=no)
options.emgDataYesNo = if_dne(options,'emgDataYesNo',0);

% Specify these if .lvm columns are not properly labeled. They say which
% column in the .lvm file contains the relavent data.
% Tap input signal
% options.tapperColumns = if_dne(options,'tapperColumns',5);
% Loading data
% options.loadColumns = if_dne(options,'loadColumns',[]);
% Positional data
% options.posColumns = if_dne(options,'posColumns',[]);
% EMG data
% options.emgColumns = if_dne(options,'emgColumns',[]);
% Sample rate for LabView collection
% options.lvmSampleRate = if_dne(options,'lvmSampleRate',50000);

% Should results plots be created? (1=yes, 0=no)
options.plotYesNo = if_dne(options,'plotYesNo',0);
options.figNumber = if_dne(options,'figNumber',1001); % figure number for 
    % wave speed plots; frequency spectra plots appear on figNumber + 1;
    % figs 1001, 1002 (defaults) will be cleared every time the code is run

% Common if_dne options that are only used when analyzing laser vibrometer
% data

% Shift data in time to account for hardware delay in laser 
% vibrometers? (1=yes, 0=no)
options.timeShiftDataYesNo = if_dne( options, 'timeShiftDataYesNo', 0 ) ;
% For Polytec PDV-100 lasers hardware delay is 1.1 msec and varies
% <1 nsec between lasers
options.timeShift = if_dne( options, 'timeShift', 0 ) ;
% Take derivative of velocity data to compute non-contact
% acceleration data? (1=yes, 0=no)
options.takeDerivYesNo = if_dne( options, 'takeDerivYesNo', 0 ) ;
% takeDerivYesNo option useful when analyzing a short transient
% response because it increases the curvature of the waves for
% improved performance of normxcorr2

switch options.collectionMethod
    case { 'accelerometer' }
        % Was accelerometer data collected? (1=yes, 0=no)
        options.accDataYesNo = 1;
        % Was the input tap signal collected? (1=yes, 0=no)
        options.tapDataYesNo = 1;
        % How many accelerometers were used?
        options.numAcc = if_dne(options,'numAcc',2);
        % What order were the accelerometers in, closest to farthest from
        % tapper?
        options.measOrder = if_dne(options,'measOrder',[1:options.numAcc]);
        % Were any of the accelerometers facing the wrong direction? If so,
        % place a negative one in the vector below corrsponding to the
        % relavent accelerometer
        options.signCorrection = if_dne(options,'signCorrection',...
            ones(1,length(options.measOrder)));
        % How far between accelerometers
        options.travelDist = if_dne(options,'travelDist',10);
        % specify these if .lvm columns are not properly labeled
        % options.accColumns = if_dne(options,'accColumns',[2:4]);
        
        % What exponent should be applied to the accelerometer data? This
        % increases the peakedness of the data to improve the performance
        % of normxcorr2.
        options.signalExp = if_dne( options, 'signalExp', 2 ) ;
               
    case 'laser'
        % -----------------------------------------------------------------
        % Note: To minimize syntax changes, still using acc label, but
        % using new case to allow for settings necessary for processing
        % laser vibrometer data
        % !!!TODO!!! propagate laser collection method throughout functions
        % -----------------------------------------------------------------
        
        % Shift data in time to account for hardware delay in laser 
        % vibrometers? (1=yes, 0=no)
        options.timeShiftDataYesNo = 1 ;
        % For Polytec PDV-100 lasers hardware delay is 1.1 msec and varies
        % <1 nsec between lasers
        options.timeShift = 1.1 ;
        % Take derivative of velocity data to compute non-contact
        % acceleration data? (1=yes, 0=no)
        options.takeDerivYesNo = if_dne( options, 'takeDerivYesNo', 0 ) ;
        % takeDerivYesNo option useful when analyzing a short transient
        % response because it increases the curvature of the waves for
        % improved performance of normxcorr2
        
        % -----------------------------------------------------------------
        % copied from accelerometer case above
        % ------------------------------------

        % Was accelerometer data collected? (1=yes, 0=no)
        options.accDataYesNo = 1;
        % Was the input tap signal collected? (1=yes, 0=no)
        options.tapDataYesNo = 1;
        % How many accelerometers were used?
        options.numAcc = if_dne(options,'numAcc',2);
        % What order were the accelerometers in, closest to farthest from
        % tapper?
        options.measOrder = if_dne(options,'measOrder',[1:options.numAcc]);
        % Were any of the accelerometers facing the wrong direction? If so,
        % place a negative one in the vector below corrsponding to the
        % relavent accelerometer
        options.signCorrection = if_dne(options,'signCorrection',...
            ones(1,length(options.measOrder)));
        % How far between accelerometers
        options.travelDist = if_dne(options,'travelDist',10);
        % specify these if .lvm columns are not properly labeled
        % options.accColumns = if_dne(options,'accColumns',[2:4]);
        
        % What exponent should be applied to the accelerometer data? This
        % increases the peakedness of the data to improve the performance
        % of normxcorr2.
        options.signalExp = if_dne( options, 'signalExp', 2 ) ;
        % -----------------------------------------------------------------
        
    case 'ultrasound'
        options.accDataYesNo = 0;
        options.tapDataYesNo = if_dne(options,'tapDataYesNo',1);
%         options.collectionTime = if_dne(options,'collectionTime',2); % [s]
        % Will tendon boundaries be determined manually or automatically?
        options.findTendon = if_dne(options,'findTendon','auto');
%         options.findTendon = if_dne(options,'findTendon','manual');
        % Was the transducer flipped so that the lower numbered collection
        % line was further from the tapper? (1=yes, 0=no)
        options.transducerFlip = if_dne(options,'trandsucerFlip',1);
        % What was the length of the transducer in mm?
        options.transducerLength = if_dne(options,'transducerLength',38);
%         options.transducerLength = if_dne(options,'transducerLength',20);
        % Was data collected that will sync the ultrasound collection with
        % another collection?
        options.syncDataYesNo = if_dne(options,'syncDataYesNo',0);
%         options.syncColumns = if_dne(options,'syncColumns',[]);
end

% How long between piezo extension and retraction before tap singal is no
% longer considered a pulse. Distinguishes between pulse and 50% duty cycle
% trials (generally). Chosen arbitrarily, but can handle our current cases
% (ms)
options.pulseThresh = if_dne(options,'pulseThresh',3);

switch options.waveSpeedMethod
    case 'XCorr'
        % What band should be used for bandpass filtering of the wave data
        % prior to wave speed calculation?
        options.filterBandWave = if_dne(options,'filterBandWave',[150 5000]);
        % For cross-correlation, what is the max delay that should be
        % considered between the wave passing subsequent measurement
        % locations?
        options.maxDelay = if_dne(options,'maxDelay',1); % [ms]
        % How much data after the tap event should be used for
        % cross-correlation?
        options.window = if_dne(options,'window',[0 3]); % [ms]
%         options.window = if_dne(options,'window','adaptive1'); % data through 1st peak
%         options.window = if_dne(options,'window','adaptive2'); % data through 2nd peak
        % Do you want to use the mex version of normxcorr2 (faster)?
        options.normxcorr2_mex = if_dne(options,'normxcorr2_mex',0); % 1 for yes
        % What is the minimum acceptable correlation between wave data at
        % subsequent locations?
        options.minCorr = if_dne(options,'minCorr',0);
        options.plotCorr = if_dne(options,'plotCorr',0);
        
    case 'P2P'
        % What band should be used for bandpass filtering of the wave data
        % prior to wave speed calculation?
        options.filterBandWave = if_dne(options,'filterBandWave',[150 5000]);
        % For peak finding, what is the max delay that should be
        % considered between the wave passing subsequent measurement
        % locations?
        options.maxDelay = if_dne(options,'maxDelay',1); % [ms]
        % How much data after the tap event should be searched for peaks?
        options.window = if_dne(options,'window',[0 3]); % [ms]
%         options.window = if_dne(options,'window','adaptive1'); % data through 1st peak
%         options.window = if_dne(options,'window','adaptive2'); % data through 2nd peak
        % Which peaks are of interest? Numbered from start of window.
        options.whichPeaks = if_dne(options,'whichPeaks',[]);
        % At least how far apart are peaks of intrest?
        options.minPeakSeparation = if_dne(options,'minPeakSeparation',1.25); % [ms]
        
    case 'frequency'
        % How should peaks in the vibration frequency spectra be chosen?
        options.peakFindMethod = if_dne(options,'peakFindMethod','manual');
%         options.peakFindMethod = if_dne(options,'peakFindMethod','auto');
        % How long was the tendon (between nodes, e.g., grip-to-grip)
        options.tendonLength = if_dne(options,'tendonLength',70); % [mm]
        % What band should be used for bandpass filtering of the wave data
        % prior to wave speed calculation?
        options.filterBandWave = if_dne(options,'filterBandWave',[150 2000]);
end

% How much can wave speed change between frames without being considered
% erroneous data
options.deltaWSThresh = if_dne(options,'deltaWSThresh',25); % [m/s]
% How long can wave speed data segments be between NaNs before they are
% thrown out?
options.minSegLength = if_dne(options,'minSegLength',3); % [frames]
% Should NaN wave speed data be filled in? (1 for yes)
options.nanFill = if_dne(options,'nanFill',0);
% How should wave speed and other data be filtered?
options.filterLowPass = if_dne(options,'filterLowPass',12);

optionsOut = options;
clear options
function [varOut] = if_dne(options,opt,newValue)
% This function initializes and defines a variable only if it was
% previously undefined. If it was previously defined, its original value
% will be kept. Use only for structure subfields, where main structure has
% been made global.
% 
% varName is a string matching the variable to be kept or redefined
% newValue can be numeric or can be a string
% 
% Jack Martin, UW-NMBL, 05/25/2017

try varOut = options.(opt);
catch
    varOut = newValue;
end


function [filenames,data] = wsFileImport(filenames,options)
% This function imports files specified by the user or allows a user to
% select files by hand.
%
% Jack Martin - 05/25/17

switch options.collectionMethod
    case { 'accelerometer', 'laser' }
        try
            filenames.lvm;
        catch
            if strcmp(options.fileSelectionMode,'choose')
                % User selects file from default directory
                disp('Select .lvm file')
                [f,d] = uigetfile(...
                    [options.defaultDirectory '\' '*.lvm'],'Select .lvm file');
            elseif strcmp(options.fileSelectionMode,'recent')
                try
                    % Last file saved to default directory
                    d = options.defaultDirectory;
                    
                    fileList = dir([d,filesep,'*.lvm']);
                    fileList = fileList(3:end);
                    saveTime = [fileList(:).datenum].';
                    [~,saveTimeInd] = sort(saveTime,'descend');
                    sortedFiles = {fileList(saveTimeInd).name};
                    f = sortedFiles{1};
                catch 
                    disp(strvcat({'No .lvm files in default directory',...
                        'try specifying options.defaultDirectory, or...'}))
                    
                    % User selects file from default directory
                    disp('Select .lvm file')
                    [f,d] = uigetfile(...
                        [options.defaultDirectory '\' '*.lvm'],'Select .lvm file');
                end
            end
            
            filenames.lvm = [d '\' f];
        end
        data.lvm = wsReadLVM(filenames.lvm,options);
        
        if options.loadDataYesNo
            switch options.loadDataLoc
                case 'mat'
                    try
                        filenames.mat;
                    catch
                        disp('Select .lvm file')
                        [f,d] = uigetfile(...
                            [options.defaultDirectory '\' '*.lvm'],...
                            'Select .lvm file');
                        filenames.lvm = [d '\' f];
                    end
                    data.mat = wsReadMAT_bertec(filenames.mat,options);
            end
        end
        
    case 'ultrasound'
        try
            filenames.mat;
        catch
            disp('Select .mat file')
            [f,d] = uigetfile(...
                [options.defaultDirectory '\' '*.mat'],'Select .mat file');
            filenames.mat = [d '\' f];
        end
        data.mat = wsReadMAT_US(filenames.mat,options);
        
        if options.loadDataYesNo
            try
                filenames.lvm;
            catch
                disp('Select .lvm file')
                [f,d] = uigetfile(...
                    [options.defaultDirectory '\' '*.lvm'],'Select .lvm file');
                filenames.lvm = [d '\' f];
            end
            data.lvm = wsReadLVM(filenames.lvm,options);
        end
end
function [lvmData] = wsReadLVM(filenameLVM,options)
% Reads wave data and ancillary data from .lvm file.
% filenameLVM must include the directory and extension, and referenced file
% must be tab-delimited
%
% Jack Martin - 05/25/17
% Based on code written by Scott Brandon

% check that file is a .lvm file
if ~strcmp(filenameLVM(end-3:end),'.lvm')
    errordlg('Selected file is not a .lvm file; try again')
    return
end

% Was this an ultrasound trial?
switch options.collectionMethod
    case 'ultrasound'
        ult = 1;
    otherwise
        ult = 0;
end

% Shorthand for which data were collected
ayn = options.accDataYesNo;
tyn = options.tapDataYesNo;
lyn = options.loadDataYesNo;
pyn = options.posDataYesNo;
eyn = options.emgDataYesNo;
if lyn, switch options.loadDataLoc, case 'lvm', here = 1;
    otherwise, here = 0; end; end
if ult, syn = options.syncDataYesNo; end

fid = fopen(filenameLVM);
% read 1st 100 lines of file to determine number of header lines
lines = cell(100,1);
for i=1:100
    lines{i,1} = fgetl(fid);
end

% determine number of header lines
endOfHeader = find(~cellfun(@isempty,strfind(lines,'End_of_Header')));
try
    nHeaderLines = endOfHeader(end) + 1;
catch
    nHeaderLines = 0;
end

fclose(fid);

% import data from file
[dataStruc] = importdata(filenameLVM,'\t',nHeaderLines);
if nHeaderLines > 0
    data = dataStruc.data;
    columnLabels = strsplit(dataStruc.textdata{end});
    
    % Find columns for each type of data
    if ayn, accCol = find(~cellfun(@isempty,strfind(columnLabels,'Accel'))); end
    if tyn, tapperCol = find(~cellfun(@isempty,strfind(columnLabels,'Tapper'))); end
    if lyn, if here, loadCol = find(~cellfun(@isempty,strfind(columnLabels,'Load'))); end; end
    if pyn, posCol = find(~cellfun(@isempty,strfind(columnLabels,'Pos'))); end
    if eyn, emgCol = find(~cellfun(@isempty,strfind(columnLabels,'EMG'))); end
    if ult, if syn, syncCol = find(~cellfun(@isempty,strfind(columnLabels,'Sync'))); end; end
else
    data = dataStruc;
    columnLabels = [];
    if ayn, accCol = []; end
    if tyn, tapperCol = []; end
    if lyn, if here, loadCol = []; end; end
    if pyn, posCol = []; end
    if eyn, emgCol = []; end
    if ult, if syn, syncCol = []; end; end
end

if ishandle(999)
    close(999)
end

% Using user-defined column numbers if labels are incompatible
if ayn
    try % use user-defined column numbers
        accCol = options.accColumns;
    catch % allow user to define columns now based on plot
        if isempty(accCol) % If columns don't have standard labels
            legendEntries = cell(1,size(data,2));
            
            for i = 1 : length(legendEntries)
                legendEntries{i} = ['Column ' num2str(i)];
            end
            
            figure(999),plot(data),legend(legendEntries)
            
            accCol = input(['Input vector of accelerometer column numbers \n'...
                'use Fig. 999 for reference \n'...
                '(can instead specify options.accColumns) \n\n']);
        end
    end
else
    numAcc = options.numAcc;
    measOrder = options.measOrder;
%     accCol = accCol(measOrder(1:numAcc))
    % done in processRawData
end

if tyn
    try % use user-defined column numbers
        tapperCol = options.tapperColumns;
    catch % allow user to define columns now based on plot
        if isempty(tapperCol) % If columns don't have standard labels
            legendEntries = cell(1,size(data,2));
            
            for i = 1 : length(legendEntries)
                legendEntries{i} = ['Column ' num2str(i)];
            end
            
            if ~ishandle(999)
                figure(999),plot(data),legend(legendEntries)
            else
                figure(999)
            end
            
            tapperCol = input(['Input tapper column number \n'...
                'use Fig. 999 for reference \n'...
                '(can instead specify options.tapperColumns) \n\n']);
        end
    end
end

if lyn, if here
    try % use user-defined column numbers
        loadCol = options.loadColumns;
    catch % allow user to define columns now based on plot
        if isempty(loadCol) % If columns don't have standard labels
            legendEntries = cell(1,size(data,2));
            
            for i = 1 : length(legendEntries)
                legendEntries{i} = ['Column ' num2str(i)];
            end
            
            if ~ishandle(999)
                figure(999),plot(data),legend(legendEntries)
            else
                figure(999)
            end
            
            loadCol = input(['Input vector of load column numbers \n'...
                'use Fig. 999 for reference \n'...
                '(can instead specify options.loadColumns) \n\n']);
        end
    end
end,end

if pyn
    try % use user-defined column numbers
        posCol = options.posColumns;
    catch % allow user to define columns now based on plot
        if isempty(posCol) % If columns don't have standard labels
            legendEntries = cell(1,size(data,2));
            
            for i = 1 : length(legendEntries)
                legendEntries{i} = ['Column ' num2str(i)];
            end
            
            if ~ishandle(999)
                figure(999),plot(data),legend(legendEntries)
            else
                figure(999)
            end
            
            posCol = input(['Input vector of position column numbers \n'...
                'use Fig. 999 for reference \n'...
                '(can instead specify options.posColumns) \n\n']);
        end
    end
end

if eyn
    try % use user-defined column numbers
        emgCol = options.emgColumns;
    catch % allow user to define columns now based on plot
        if isempty(emgCol) % If columns don't have standard labels
            legendEntries = cell(1,size(data,2));
            
            for i = 1 : length(legendEntries)
                legendEntries{i} = ['Column ' num2str(i)];
            end
            
            if ~ishandle(999)
                figure(999),plot(data),legend(legendEntries)
            else
                figure(999)
            end
            
            emgCol = input(['Input vector of EMG column numbers \n'...
                'use Fig. 999 for reference \n'...
                '(can instead specify options.emgColumns) \n\n']);
        end
    end
end

if ult, if syn
    try % use user-defined column numbers
        syncCol = options.syncColumns;
    catch % allow user to define columns now based on plot
        if isempty(syncCol) % If columns don't have standard labels
            legendEntries = cell(1,size(data,2));
            
            for i = 1 : length(legendEntries)
                legendEntries{i} = ['Column ' num2str(i)];
            end
            
            if ~ishandle(999)
                figure(999),plot(data),legend(legendEntries)
            else
                figure(999)
            end
            
            syncCol = input(['Input vector of sync column numbers \n'...
                'use Fig. 999 for reference \n'...
                '(can instead specify options.syncColumns) \n\n']);
        end
    end
end,end

% Extract data
% ------------

% header/labels
if nHeaderLines > 0
    % determine sample rate
    dxRow = ~cellfun(@isempty,strfind(dataStruc.textdata,'Delta_X'));
    %sample rate row is labeled "Delta_x"
    dxDat = strsplit(dataStruc.textdata{dxRow});
    sampleRateIndi = 1./cell2mat(cellfun(@str2num,dxDat,'uniformoutput',0));
    %sample rate for each channel
    lvmData.sampleRate = sampleRateIndi(1);
    % Hz, assuming column 1 was time (true); assumes constant sample rate across channels
    
    % extract labels
    if ayn, lvmData.labels.wave = columnLabels(accCol); end
    if tyn, lvmData.labels.tapper = columnLabels(tapperCol); end
    if lyn, if here, lvmData.labels.load = columnLabels(loadCol); end, end
    if pyn, lvmData.labels.pos = columnLabels(posCol); end
    if eyn, lvmData.labels.emg = columnLabels(emgCol); end
    if ult, if syn, lvmData.labels.sync = columnLabels(syncCol); end, end
else
    try
        lvmData.sampleRate = options.lvmSampleRate;
    catch
        lvmData.sampleRate = input(['Input sample rate for LVM data [Hz] \n'...
            '(can instead specify options.lvmSampleRate) \n\n']);
    end
end

% sensor data
% if using laser vibrometers, then need to time shift data to account for
% hardware delay
if options.timeShiftDataYesNo
    try
        timeShift = round( lvmData.sampleRate * ( options.timeShift / 1000 ), 0 ) ;
    catch
        timeShift = 0 ;
    end
else
    timeShift = 0 ;
end

if ayn, lvmData.wave = data( 1 + timeShift : end, accCol ) ; end
if tyn, lvmData.tapper = data( 1 : end - timeShift, tapperCol ) ; end
if lyn, if here, lvmData.load = data( 1 : end - timeShift, loadCol ) ; end, end
if pyn, lvmData.pos = data( 1 : end - timeShift, posCol ) ; end
if eyn, lvmData.emg = data( 1 : end - timeShift, emgCol ) ; end
if ult, if syn, lvmData.sync = data( 1 : end - timeShift, syncCol ) ; end, end
function [matData] = wsReadMAT_bertec(filenameMAT,options)
% Reads loading data from .mat file written by bertec force plate. This
% data collection should be longer than the .lvm collection, and should
% overlap the .lvm data at both ends.
% filenameMAT must include the directory and extension
%
% Jack Martin - 06/09/17

% check that file is a .mat file
if ~strcmp(filenameMAT(end-3:end),'.mat')
    errordlg('Selected file is not a .mat file; try again')
    return
end

bertec = load(filenameMAT);
matData.sampleRate = 1/mean(diff(bertec.data(1,:))); % [Hz]
matData.load = bertec.data(2,:)'; % [N]
function [matData] = wsReadMAT_US(filenameMAT,options)
% Reads wave data from .mat file.
% filenameMAT must include the directory and extension
%
% Jack Martin - 06/09/17

% check that file is a .mat file
if ~strcmp(filenameMAT(end-3:end),'.mat')
    errordlg('Selected file is not a .mat file; try again')
    return
end

load(filenameMAT)
rf = rfd.rf;
rfUpsamp = stp.yur;
dx = stp.dv;

try
    timeSec = options.collectionTime;
catch
    timeSec = input(['Input ultrasound collection time [s] \n'...
        '(can instead specify options.collectionTime) \n\n']);
end

if mod(size(stp.dv,3),2)~=0
    dx = dx(:,:,1:(end-1));
end

framerate = size(stp.dv,3)/timeSec;
frameDelay = 1/size(stp.dv,2);
col = flipud([1 0 0; 0.8 0.8 0; 0 0.7 0; 0 0.8 0.8; 0 0 1]);

% Determine section of interest within RF data
for i = 1 : size(rf,2)
    switch options.findTendon
        case 'auto'
            % Automatic selection of region of interest based on first
            % frame of RF data. Works when we don't expect much motion. Can
            % also select tendon section of interest on a frame-by-frame
            % basis as Scott Brandon had done in
            % "extractUltrasoundWaves_v2.m"
            
            % Rectifying 1st frame of data
            rfRect = abs(rf(:,i,1));
            % Set 1st half of data to zero to get rid of transducer edge
            rfRect(1:round(length(rfRect)/2)) = 0;
            % Apply filter to smooth edges of RF data
            cutoff = 10;
            Wn = cutoff/(size(rf,1)/timeSec/2);
            [bf,af] = butter(2,Wn);
            rfFilt = filtfilt(bf,af,rfRect);
            % Find points above a certain threshold
            thresh = max(rfFilt)*0.4;
            tL(1) = find(rfFilt>thresh,1,'first');
            tL(2) = find(rfFilt>thresh,1,'last');
            tendonLims(i,:) = tL;
        case 'manual'
            % Manual selection of region of interest
            figure(746)
            % Plot first frame of RF data
            plot(rf(:,i,1),'color',col(i,:))
            title('Select minimum and maximum ROI depth')
            disp('Select minimum and maximum ROI depth in Fig. 746')
            tL = ginput(2);
            tendonLims(i,:) = tL(:,1);
            close(746)
    end
    
    % Extract data from section of interest
    jdepth{i} = [tendonLims(i,1):25:tendonLims(i,2)]*rfUpsamp;
    
    for j = 1 : length(jdepth{i}) % for each kernel
        if ~isempty(find(stp.pts > jdepth{i}(j),1,'first'))
            jj{i}(j)= find(stp.pts > jdepth{i}(j),1,'first');
        else
            break;
        end
    end
    jjL(i) = length(jj{i});
end

% Take same number of kernels from all line locations
jjLmin = min(jjL);
for i = 1 : size(rf,2)
    if length(jj{i}) > jjLmin
        jj{i} = jj{i}(1:jjLmin);
    end
end

% Extract incremental displacements
for i = 1 : size(dx,2) % for each measurement line
    for j = 1 : length(jj{i}) % for each kernel
        DX(:,j,i) = dx(jj{i}(j),i,:);
        vel(:,j,i) = DX(:,j,i)/1000/(1/framerate); % [m/s]
    end
end

% Interpolating data and accounting for frame shift
upSamp = 1/frameDelay;
if upSamp > 1
    for i = 1 : size(dx,2)
        for j = 1 : length(jj{i})
            Vel(:,j,i) = interp1(1:size(vel,1),vel(:,j,i),1:0.5:size(vel,1)+0.5,'spline');
            VelShift(:,j,i) = [Vel([(1+upSamp-i):(end-(i-1))],j,i);...
                zeros(upSamp-1,1)];
            velShift(:,j,i) = VelShift(1:2:end-1,j,i);
        end
    end
end

% Flipping order of data if transducer was backwards
if options.transducerFlip
    vel = flip(velShift,3);
end

% Calculating mean velocity across depth
for i = 1 : size(dx,2)
    velMean(:,i) = mean(vel(:,:,i),2);
end

matData.sampleRate = framerate;
matData.collectionTime = timeSec;
matData.wave = velMean;
matData.waves = vel;

function [tapTiming] = wsTapTiming(rawData,options)
% Creates indices to split wave data based on tap timing. Deterimines tap
% rate and duty cycle.

try
    tapSig = rawData.lvm.tapper;
    
    % Get rid of any aberrant negative values
    for i = 2 : length(tapSig)-1
        if tapSig(i) < 0
            tapSig(i) = (tapSig(i-1)+tapSig(i+1))/2;
        end
    end
    
    % find pulse edges
    extended = tapSig > max(tapSig)/2;
    leading = find([0; diff(extended) > 0] > 0);
    retracted = tapSig < max(tapSig)/2;
    trailing = find([0; diff(retracted) > 0] > 0);
    edges = sort([leading; trailing]);
    
    sampleRate = rawData.lvm.sampleRate;
    
    % Determining tap rate (Hz)
    tapTiming.tapRate = round(sampleRate/mean(diff(leading)));
    
    % Determining duty cycle (fractional)
    if length(leading) > length(trailing)
        tapTiming.dutyCycle = ...
            mean(trailing - leading(1:end-1))/mean(diff(leading));
    elseif length(leading) < length(trailing)
        tapTiming.dutyCycle = ...
            mean(trailing(2:end) - leading)/mean(diff(leading));
    elseif length(leading) == length(trailing)
        if leading(1) < trailing(1)
            tapTiming.dutyCycle = ...
                mean(trailing - leading)/mean(diff(leading));
        elseif leading(1) > trailing(1)
            tapTiming.dutyCycle = ...
                mean(trailing(2:end) - leading(1:end-1))/mean(diff(leading));
        end
    end
    
    tapTiming.tapDuration = 1000/tapTiming.tapRate*tapTiming.dutyCycle; %[ms]
    
    % Accommodating negative window bounds
    if isfloat(options.window)
        if options.window(1) < 0
            windowStart = round(options.window(1)*sampleRate/1000);
            edges = edges + windowStart;
            leading = leading + windowStart;
            trailing = trailing + windowStart;
            
            if edges(1) <= 0, edges = edges(2:end); end
            if leading(1) <= 0, leading = leading(2:end); end
            if trailing(1) <= 0, trailing = trailing(2:end); end
        end
    end
    
    tapTiming.edges = edges;
    tapTiming.leading = leading;
    tapTiming.trailing = trailing;
catch
    % Define tap timing manually
    switch options.collectionMethod
        case 'accelerometer'
            sampleRate = rawData.lvm.sampleRate;
            wv = rawData.lvm.wave;
        case 'ultrasound'
            sampleRate = rawData.mat.sampleRate;
            wv = rawData.mat.wave;
    end
    
    try
        tapRate = options.tapRate;
    catch
        tapRate = input(['Input tap rate [Hz] \n'...
            '(can instead specify options.tapRate) \n\n']);
    end
    
    % Plot section of data from begininng of collection and prompt user to
    % define first tap
    threeTaps = round(3*sampleRate/tapRate);
    figure(837)
    plot(wv(1:threeTaps,:))
    title({'Click a location just prior to first full wave'...
        'Make sure to click before the start of the wave'})
    disp(strvcat({'Click a location just prior to first full wave in Fig. 837',...
        'Make sure to click before the start of the wave'}))
    [tapInd,~] = ginput(1);
    tapInd = floor(tapInd);
    close(837)
    
    % Define tap timing based on user selection and known tap rate
    try
        timeSec = options.collectionTime;
    catch
        timeSec = input(['Input ultrasound collection time [s] \n'...
            '(can instead specify options.collectionTime) \n\n']);
    end
    
    tapInterval = sampleRate/tapRate;
    numTapsApprox = ceil(timeSec*tapRate);
    tapInds = tapInd + round(tapInterval*[0:numTapsApprox-1]);
    
    % Accommodating negative window bounds
    if isfloat(options.window)
        if options.window(1) < 0
            windowStart = options.window(1)*sampleRate/1000;
            tapInds = tapInds + windowStart;
        end
    end
    
    tapTiming.tapRate = tapRate;
    tapTiming.tapInds = tapInds(tapInds < timeSec*sampleRate);
end


function [params] = windowBounds(rawData,options,params)
% Determines window indices for static windows. This function was created
% primarily to deal with negative window bounds.
%
% Jack Martin - 11/07/18

if isfloat(options.window)
    switch options.collectionMethod
        case 'accelerometer'
            sampleRate = rawData.lvm.sampleRate;
        case 'ultrasound'
            sampleRate = rawData.mat.sampleRate;
    end
        
    windowTime = options.window;
    windowBnds = round(options.window/1000*sampleRate) + 1;
    windowInds = windowBnds(1) : windowBnds(2);
    
    if windowTime(1) < 0
        % Shifting window bounds. The negative window is accounted for in
        % wsTapTiming, which shifts the location of tap edges
        windowInds = windowInds - windowInds(1) + 1;
    end
    
    params.windowInds = windowInds;
end


function [processed,params] = ...
    wsProcessRawData(rawData,options,params,filenames)
% Prepares wave data for wave speed calculation. Corrects sign/order
% errors, crops, filters, splits into sections based on tap timing.
% 
% Jack Martin - 05/25/17

% Shorthand for which data were collected
lyn = options.loadDataYesNo;
pyn = options.posDataYesNo;
eyn = options.emgDataYesNo;
tyn = options.tapDataYesNo;

% Extracting parameters
try sampleRate = rawData.lvm.sampleRate;
    catch, sampleRate = rawData.mat.sampleRate; end
filterBandWave = options.filterBandWave;
filterLowPass = options.filterLowPass;

% Extracting ancillary data
if lyn, switch options.loadDataLoc
    case 'lvm'
        Load = rawData.lvm.load;
    case 'mat'
        LoadMAT = rawData.mat.load;
        sampleRateLoad = rawData.mat.sampleRate;
        upSampLoad = round(sampleRate/sampleRateLoad);
        Load = interp(LoadMAT,upSampLoad);
end, end

if pyn, pos = rawData.lvm.pos; end
if eyn, emg = rawData.lvm.emg; end

if tyn
    tapSig = rawData.lvm.tapper;
    
    % Get rid of any aberrant negative values
    for i = 2 : length(tapSig)-1
        if tapSig(i) < 0
            tapSig(i) = (tapSig(i-1)+tapSig(i+1))/2;
        end
    end
    
    pulseThresh = options.pulseThresh;
    leading = params.tapTiming.leading; leadingWave = leading;
    trailing = params.tapTiming.trailing; trailingWave = trailing;
    edges = params.tapTiming.edges; edgesWave = edges;
    tapRate = params.tapTiming.tapRate;
    tapDuration = params.tapTiming.tapDuration; % [ms]
    dutyCycle = params.tapTiming.dutyCycle;
else
    tapInds = params.tapTiming.tapInds;
end

% Extracting wave data and defining collection mode dependent vars/params
switch options.collectionMethod
    case { 'accelerometer', 'laser' }
        waveRaw = rawData.lvm.wave;
        signCorr = options.signCorrection;
        waveRaw = waveRaw.*signCorr; % Correct accelerometer direction errors
        measOrder = options.measOrder;
        waveRaw(:,measOrder) = waveRaw; % Correct accelerometer order
%         errors, was done both here and readLVM
        sampleRateWave = sampleRate;
        
    case 'ultrasound'
        waveRaw = rawData.mat.wave;
        measOrder = 1:size(waveRaw,2);
        if options.transducerFlip
            measOrder = fliplr(measOrder);
        end
        sampleRateWave = rawData.mat.sampleRate;
        collectionTime = rawData.mat.collectionTime;
        syn = options.syncDataYesNo;
        % If there is ultrasound sync data, crop other signals accordingly
        if syn, Sync = rawData.lvm.sync;
            % Finding when ultrasound is recording
            syncSmooth = smooth(Sync,9);            
            USstart = find(syncSmooth > 0.5,1,'first');
            USend = USstart + round(collectionTime*sampleRate) - 1;
            
            % Cropping to active US period,
            % and interpolating to match US sample rate
            USinds = USstart:USend-1;
            xInterp = [1:size(waveRaw,1)]*sampleRate/sampleRateWave;
            
            if lyn, Load = Load(USinds,:); end
            if pyn, pos = pos(USinds,:); end
            if eyn, emg = emg(USinds,:); end
            if tyn
                tapSig = tapSig(USinds,:);
                leadingWave = leading(leading > USstart);
                leadingWave = leadingWave(leadingWave < USend) - USstart + 1;
                trailingWave = trailing(trailing > USstart);
                trailingWave = trailingWave(trailingWave < USend) - USstart + 1;
                edgesWave = edges(edges > USstart);
                edgesWave = edgesWave(edgesWave < USend) - USstart + 1;
            else
                tapIndsWave = tapInds;
                tapInds = round(tapInds*sampleRate/sampleRateWave);
            end
        else
            if tyn
                leadingWave = leading;
                trailingWave = trailing;
                edgesWave = edges;
            else
                tapIndsWave = tapInds;
                tapInds = round(tapInds*sampleRate/sampleRateWave);
            end
        end
        
        % Determining US line locations, travel distance
        underscore = strfind(filenames.mat,'_');
        USmode = filenames.mat(underscore(end)+1:end-4);
        [lineLocs,travelDist] = ...
            US_lineSpacing(USmode,options.transducerLength,options.transducerFlip);
        params.lineLocations = lineLocs;
        params.travelDist = travelDist;
end

% Filter data
WnWave = filterBandWave/(sampleRateWave/2);
[bfWave,afWave] = butter(2,WnWave);
for i = 1 : length(measOrder)
    %enables the ability to take the derivative of the raw laser data to
    %increase the curvature of the signals and improve the performance of
    %the normxcorr2
    if options.takeDerivYesNo
        waveRaw( :, i ) = gradient( waveRaw( :, i ), 1 / sampleRateWave ) ;
    end
    waveFilt(:,i) = filtfilt(bfWave,afWave,waveRaw(:,i));
end

WnLP = filterLowPass/(sampleRate/2);
[bfLP,afLP] = butter(2,WnLP,'low');

if lyn
    switch options.loadDataLoc, case 'mat'
        % First, synchronize externally collected load data
        loadCropInds = wsImpulseSync(Load,waveRaw,sampleRate);
        Load = Load(loadCropInds);
    end
    
    loadFilt = filtfilt(bfLP,afLP,Load);
end

if pyn, posFilt = filtfilt(bfLP,afLP,pos); end
if eyn, emgRect = abs(emg);
    for i = 1 : size(emg,2)
        emgRectNorm(:,i) = emgRect(:,i)/max(emgRect(:,i)); end, end

% Crop data based on tap signal
if tyn
    if tapDuration > pulseThresh % Treat push and release as separate events
        % Crop data
        waveFilt = waveFilt(edgesWave(1):edgesWave(end),:);
        tapSigCrop = tapSig(edges(1):edges(end),:);
        if lyn, loadFilt = loadFilt(edges(1):edges(end),:); end
        if pyn, posFilt = posFilt(edges(1):edges(end),:); end
        if eyn, emgRectNorm = emgRectNorm(edges(1):edges(end),:); end
        
        leadingCrop = leading - edges(1) + 1;
        leadingWaveCrop = leadingWave - edgesWave(1) + 1;
        trailingCrop = trailing - edges(1) + 1;
        trailingWaveCrop = trailingWave - edgesWave(1) + 1;
        edgesCrop = edges - edges(1) + 1;
        edgesWaveCrop = edgesWave - edgesWave(1) + 1;
        
        % Account for any emg delay
        try emgDelay = round(sampleRate/1000*options.emgDelay);
            emgRectNorm = [emgRectNorm(1+emgDelay:end,:); NaN(emgDelay,size(emgRectNorm,2))]; end
        
        % Define data segments
        m = 0;
        n = 0;
        for i = 1 : length(edgesCrop) - 1
            if sum(leadingCrop == edgesCrop(i))
                m = m + 1;
                wavePush{m,:} = detrend(...
                    waveFilt(edgesWaveCrop(i):edgesWaveCrop(i+1)-1,:),0);
                if lyn, loadPush{m,:} = ...
                        loadFilt(edgesCrop(i):edgesCrop(i+1)-1,:);
                        loadPushMean(m,:) = mean(loadPush{m,:}); end
                if pyn, posPush{m,:} = ...
                        posFilt(edgesCrop(i):edgesCrop(i+1)-1,:);
                        posPushMean(m,:) = mean(posPush{m,:}); end
                if eyn, emgPush{m,:} = ...
                        emgRectNorm(edgesCrop(i):edgesCrop(i+1)-1,:);
                        emgPushMean(m,:) = mean(emgPush{m,:}); end
            elseif sum(trailingCrop == edgesCrop(i))
                n = n + 1;
                waveRelease{n,:} = detrend(...
                    waveFilt(edgesWaveCrop(i):edgesWaveCrop(i+1)-1,:),0);
                if lyn, loadRelease{n,:} = ...
                        loadFilt(edgesCrop(i):edgesCrop(i+1)-1,:);
                        loadReleaseMean(n,:) = mean(loadRelease{n,:}); end
                if pyn, posRelease{n,:} = ...
                        posFilt(edgesCrop(i):edgesCrop(i+1)-1,:);
                        posReleaseMean(n,:) = mean(posRelease{n,:}); end
                if eyn, emgRelease{n,:} = ...
                        emgRectNorm(edgesCrop(i):edgesCrop(i+1)-1,:);
                        emgReleaseMean(n,:) = mean(emgRelease{n,:}); end
            end
        end
    else % Treat push and release as one event
        % Crop data
        waveFilt = waveFilt(leadingWave(1):leadingWave(end),:);
        tapSigCrop = tapSig(leading(1):leading(end),:);
        if lyn, loadFilt = loadFilt(leading(1):leading(end),:); end
        if pyn, posFilt = posFilt(leading(1):leading(end),:); end
        if eyn, emgRectNorm = emgRectNorm(leading(1):leading(end),:); end
        
        leadingCrop = leading - leading(1) + 1;
        leadingWaveCrop = leadingWave - leadingWave(1) + 1;
        trailingCrop = trailing - leading(1) + 1;
        trailingWaveCrop = trailingWave - leading(1) + 1;
        edgesCrop = edges - leading(1) + 1;
        edgesWaveCrop = edgesWave - leadingWave(1) + 1;
        
        % Account for any emg delay
        try emgDelay = round(sampleRate/1000*options.emgDelay);
            emgRectNorm = [emgRectNorm(1+emgDelay:end,:); NaN(emgDelay,size(emgRectNorm,2))]; end
    
        % Define data segments
        for i = 1 : length(leadingCrop) - 1
            wavePush{i,:} = detrend(...
                waveFilt(leadingWaveCrop(i):leadingWaveCrop(i+1)-1,:),0);
            if lyn, loadPush{i,:} = ...
                    loadFilt(leadingCrop(i):leadingCrop(i+1)-1,:);
                    loadPushMean(i,:) = mean(loadPush{i,:}); end
            if pyn, posPush{i,:} = ...
                    posFilt(leadingCrop(i):leadingCrop(i+1)-1,:);
                    posPushMean(i,:) = mean(posPush{i,:}); end
            if eyn, emgPush{i,:} = ...
                    emgRectNorm(leadingCrop(i):leadingCrop(i+1)-1,:);
                    emgPushMean(i,:) = mean(emgPush{i,:}); end
        end
        waveRelease = [];
        if lyn, loadRelease = []; loadReleaseMean = []; end
        if pyn, posRelease = []; posReleaseMean = []; end
        if eyn, emgRelease = []; emgReleaseMean = [];  end
    end
else
    % Crop data
    waveFilt = waveFilt(tapIndsWave(1):tapIndsWave(end),:);
    if lyn, loadFilt = loadFilt(tapInds(1):tapInds(end),:); end
    if pyn, posFilt = posFilt(tapInds(1):tapInds(end),:); end
    if eyn, emgRectNorm = emgRectNorm(tapInds(1):tapInds(end),:); end
    
    tapIndsCrop = tapInds - tapInds(1) + 1;
    tapIndsWaveCrop = tapIndsWave - tapIndsWave(1) + 1;
    params.tapTiming.tapIndsCrop = tapIndsWaveCrop;
    
    % Account for any emg delay
    try emgDelay = round(sampleRate/1000*options.emgDelay);
        emgRectNorm = [emgRectNorm(1+emgDelay:end,:); NaN(emgDelay,size(emgRectNorm,2))]; end
    
    % Define data segments
    for i = 1 : length(tapIndsCrop) - 1
        wavePush{i,:} = detrend(...
            waveFilt(tapIndsWaveCrop(i):tapIndsWaveCrop(i+1)-1,:),0);
        if lyn, loadPush{i,:} = ...
                loadFilt(tapIndsCrop(i):tapIndsCrop(i+1)-1,:);
                loadPushMean(i,:) = mean(loadPush{i,:}); end
        if pyn, posPush{i,:} = ...
                posFilt(tapIndsCrop(i):tapIndsCrop(i+1)-1,:);
                posPushMean(i,:) = mean(posPush{i,:}); end
        if eyn, emgPush{i,:} = ...
                emgRectNorm(tapIndsCrop(i):tapIndsCrop(i+1)-1,:);
                emgPushMean(i,:) = mean(emgPush{i,:}); end
    end
    waveRelease = [];
    if lyn, loadRelease = []; loadReleaseMean = []; end
    if pyn, posRelease = []; posReleaseMean = []; end
    if eyn, emgRelease = []; emgReleaseMean = [];  end
end

% Define outputs
processed.wave.filtered = waveFilt;
if lyn, processed.load.filtered = loadFilt; end
if pyn, processed.pos.filtered = posFilt; end
if eyn, processed.emg.rectNorm = emgRectNorm; end
if tyn
    processed.tapper.tapSig = tapSigCrop;
    params.tapTiming.edgesCrop = edgesCrop;
    params.tapTiming.leadingCrop = leadingCrop;
    params.tapTiming.trailingCrop = trailingCrop;
    params.tapTiming.edgesWave = edgesWave;
    params.tapTiming.leadingWave = leadingWave;
    params.tapTiming.trailingWave = trailingWave;
    params.tapTiming.edgesWaveCrop = edgesWaveCrop;
    params.tapTiming.leadingWaveCrop = leadingWaveCrop;
    params.tapTiming.trailingWaveCrop = trailingWaveCrop;
else
    params.tapTiming.tapIndsCrop = tapIndsCrop;
    params.tapTiming.tapIndsWave = tapIndsWave;
    params.tapTiming.tapIndsWaveCrop = tapIndsWaveCrop;
end
processed.wave.push = wavePush;
processed.wave.release = waveRelease;
if lyn, processed.load.push = loadPush;
    processed.load.pushMean = loadPushMean;
    processed.load.release = loadRelease;
    processed.load.releaseMean = loadReleaseMean; end
if pyn, processed.pos.push = posPush;
    processed.pos.pushMean = posPushMean;
    processed.pos.release = posRelease;
    processed.pos.releaseMean = posReleaseMean; end
if eyn, processed.emg.push = emgPush;
    processed.emg.pushMean = emgPushMean;
    processed.emg.release = emgRelease;
    processed.emg.releaseMean = emgReleaseMean; end
params.sampleRate.waveData = sampleRateWave;

% Define pairs of measurements to compare
if ~strcmp(options.waveSpeedMethod,'frequency')
    measNums = sort(measOrder,'ascend');
    if length(measNums) == 2
        params.measPairs = measNums;
    else
        params.measPairs = nchoosek(measNums,2);
    end
end

% Define time vectors
processed.wave.time = (0:size(waveFilt,1)-1)'/sampleRateWave*1000; % [ms]
if tyn, processed.tapper.time = (0:size(tapSigCrop,1)-1)'/sampleRate*1000; end
if lyn, processed.load.time = (0:size(loadFilt,1)-1)'/sampleRate*1000; end
if pyn, processed.pos.time = (0:size(posFilt,1)-1)'/sampleRate*1000; end
if eyn, processed.emg.time = (0:size(emgRectNorm,1)-1)'/sampleRate*1000; end
function [lineLocs,travelDist] = ...
    US_lineSpacing(USmode,transducerLength,transducerFlipped,transducerElements)
% Gives ultrasound line locations based on custom ultrasound collection
% mode (in string form). 'transducerFlipped' should equal 1 if distal end
% was towards tapper. 'transducerElements' is the number of elements.
% 
% Jack Martin - 06/16/17

switch USmode
    case '6340'
        lineNums = [63 103];
    case '2'
        lineNums = [43 103];
    case 'fill in others later'
        
    otherwise
        
end

if nargin < 4, transducerElements = 128; end

lineLocs = (lineNums-1)*transducerLength/(transducerElements-1);
if transducerFlipped, lineLocs = fliplr(transducerLength - lineLocs); end

travelDist = diff(lineLocs);
function [loadCropInds] = wsImpulseSync(Load,acc,sampleRate)
% Syncs accelerometer and load data, and crops load data accordingly. Use
% for the case of an impulive input that can be detected in both signals.
% Impulse should be largest magnitude peak in accelerometer signal.
% Load data should overlap accelerometer data at both ends, and sample
% rates should be the same.
% 
% Jack Martin - 06/14/17

filterBand = [50 500];
Wn = filterBand/(sampleRate/2);
[bf,af] = butter(2,Wn);

% Find impulse in accelerometer signal
Acc = mean(acc,2); % should occur at similar time in all signals
% Filter to be sure maximum is due to impulse, rather than tap or drift
accFilt = filtfilt(bf,af,Acc);
[~,accInd] = max(abs(accFilt));

% Find impulse in load signal
% Filter to be sure maximum is due to impulse, rather than load
loadFilt = filtfilt(bf,af,Load);
[~,loadInd] = max(abs(loadFilt));

% Find start of relevant load data
loadShift = loadInd - accInd;
loadStart = 1 + loadShift;

% Define indices for relevant load data
loadCropInds = (loadStart : loadStart + length(Acc) - 1)';


function [waveSpeed,params] = wsCalcXCorr(processedData,params,options)
% Calculates wave speed based on travel time using cross-correlation
% approach
% 
% Jack Martin - 05/30/17; Edited 05/16/18; Edited 11/7/18

wave{1} = processedData.wave.push;
if ~isempty(processedData.wave.release)
    wave{2} = processedData.wave.release;
end

measPairs = params.measPairs;
sampleRate = params.sampleRate.waveData;
maxDelay = round(options.maxDelay/1000*sampleRate);

if isfloat(options.window(1))
    windowInds = params.windowInds;
else
    if ~(strcmp(options.window,'adaptive1') || ...
            strcmp(options.window,'adaptive2'))
        disp(strvcat({'Incorrect input for options.window',...
            'Switching to default 3ms window',...
            'Try setting options.window to "adaptive1" or "adaptive2"'}))
        options.window = [0 3];
        windowInds = 1:151;
        params.windowInds = windowInds;
    end
end

try, travelDist = options.travelDist; catch, travelDist = params.travelDist; end
if length(travelDist) == 1
    travelDist = travelDist*ones(1,size(wave{1}{1},2)-1);
end

for i = 1 : length(wave) % push vs. release
    for j = 1 : size(measPairs,1) % compare waves at which locations
        for k = 1 : size(wave{i},1) % over time
            if ~isfloat(options.window) % Adaptive window mode
                testWave = wave{i}{k}(:,measPairs(j,1));
                windowInds{i,j,k} = adaptiveWindowing(testWave,options.window,sampleRate);
                params.windowInds = windowInds;
            end
            
            if iscell(windowInds)
                thisWindow = windowInds{i,j,k};
            else
                thisWindow = windowInds;
            end
            
            % Setting up template and test windows
            waveA = wave{i}{k}(thisWindow,measPairs(j,1));
            waveB = wave{i}{k}(thisWindow(1):thisWindow(end)+maxDelay-1,measPairs(j,2));
            
            % Raising waveforms to a power enhances 'peakedness', which biases
            % cross-correlation to align peaks. Keeping original sign, and
            % normalizing.
            exponent = options.signalExp;
            A = waveA.^exponent.*sign(waveA)./max(abs(waveA.^exponent));
            B = waveB.^exponent.*sign(waveB)./max(abs(waveB.^exponent));
            
            % Computing cross-correlation between signals from the two measurment
            % locations while treating the signal from the first location as the
            % template, which will be shifted relative to the signal from the
            % second location
            
            if options.normxcorr2_mex
                try
                    r = normxcorr2_mex(A,B);
                catch
                    warning('MEX version of normxcorr2 not found; switching to default')
                    r = normxcorr2(A,B);
                end
            else
                r = normxcorr2(A,B);
            end
            
            % Keeping only the valid portion of the correlation vector, i.e., where
            % there is complete overlap between the two signals
            la = length(A);
            lb = length(B);
            r = r(la:lb);
            
            rVals{i,j,k} = r;
            
            % Finding the peak correlation
            [peakCorrelation{i,j}(k),peakInd]=max(rVals{i,j,k});
            
            % Performing cosine interpolation to estimate lags with sub-frame
            % precision. See Cespedes et al., Ultrason Imaging 17, 142-171 (1995).
            if ((peakInd > 1) && (peakInd < length(r)))
                wo = acos((r(peakInd-1) + r(peakInd+1))/(2*r(peakInd)));
                theta = atan((r(peakInd-1) - r(peakInd+1))/(2*r(peakInd)*sin(wo)));
                delta = - theta/wo;
                lagFrames{i,j}(k) = peakInd - 1 + delta;
            else
                lagFrames{i,j}(k) = peakInd - 1;
            end
        end
        
        % Time lag between wave arrival at two measurement locations
        lagTime{i,j} = lagFrames{i,j}/sampleRate*1000; % [ms]
        
        % Calculating wave speed (c = deltaX/deltaT)
        wavespeed{i,j} = sum(travelDist(measPairs(j,1) : measPairs(j,2)-1))...
            ./lagTime{i,j}; % [mm/ms == m/s]
    end
end

for j = 1 : size(measPairs,1)
    waveSpeed.push{1,j} = wavespeed{1,j}';
    if ~isempty(processedData.wave.release)
        waveSpeed.release{1,j} = wavespeed{2,j}';
    else
        waveSpeed.release = [];
    end
end

params.waveCorrelation.rPeak = peakCorrelation;
params.waveCorrelation.lagFrames = lagFrames;
params.waveCorrelation.rVals = rVals;
function [windowInds] = adaptiveWindowing(wave,option,sampleRate)
% Determines window length based on locations of positive and negative
% peaks
% 
% Jack Martin - 05/17/18

peakSearchWindow{1} = round(0.2/1000*sampleRate):round(1/1000*sampleRate);
peakSearchWindow{2} = round(1/1000*sampleRate):round(2/1000*sampleRate);

[~,peakInds(1)] = max(abs(wave(peakSearchWindow{1})));
[~,peakInds(2)] = max(abs(wave(peakSearchWindow{2})));

peakInds(1) = peakInds(1) + peakSearchWindow{1}(1) - 1;
peakInds(2) = peakInds(2) + peakSearchWindow{2}(1) - 1;

if strcmp(option,'adaptive1')
    [~,nextPeakInd] = findpeaks(abs(wave(peakInds(1):end)),'npeaks',1);
    endInd = nextPeakInd + peakInds(1);
elseif strcmp(option,'adaptive2')
    [~,nextPeakInd] = findpeaks(abs(wave(peakInds(2):end)),'npeaks',1);
    endInd = nextPeakInd + peakInds(2);
end

windowInds = 1:endInd;


function [waveSpeed,params] = wsCalcP2P(processedData,params,options)
% Calculates wave speed based on travel time using peak-to-peak method.
% Could build this and XCorr method together.
% 
% Jack Martin - 10/31/17

wave{1} = processedData.wave.push;
if ~isempty(processedData.wave.release)
    wave{2} = processedData.wave.release;
end

measPairs = params.measPairs;
sampleRate = params.sampleRate.waveData;
maxDelay = round(options.maxDelay/1000*sampleRate);
peakNums = options.whichPeaks;
minPeakSeparation = round(options.minPeakSeparation/1000*sampleRate);
edges = params.tapTiming.edgesCrop;
leading = params.tapTiming.leadingCrop;

if isfloat(options.window(1))
    windowInds = params.windowInds;
else
    if ~(strcmp(options.window,'adaptive1') || ...
            strcmp(options.window,'adaptive2'))
        disp(strvcat({'Incorrect input for options.window',...
            'Switching to default 3ms window',...
            'Try setting options.window to "adaptive1" or "adaptive2"'}))
        options.window = [0 3];
        windowInds = 1:151;
        params.windowInds = windowInds;
    end
end

try, travelDist = options.travelDist; catch, travelDist = params.travelDist; end
if length(travelDist) == 1
    travelDist = travelDist*ones(1,size(wave{1}{1},2)-1);
end

for i = 1 : length(wave) % push vs. release
    for j = 1 : size(measPairs,1) % compare waves at which locations
        for k = 1 : size(wave{i},1) % over time
            if ~isfloat(options.window) % Adaptive window mode
                testWave = wave{i}{k}(:,measPairs(j,1));
                windowInds{i,j,k} = adaptiveWindowing(testWave,options.window);
                params.windowInds = windowInds;
            end
            
            if iscell(windowInds)
                thisWindow = windowInds{i,j,k};
            else
                thisWindow = windowInds;
            end
            
            waveA = wave{i}{k}(thisWindow,measPairs(j,1));
            waveB = wave{i}{k}(:,measPairs(j,2));
            
            [lagFrames{i,j}(k),peakVals{k,i,j},peakIndsRel] = ...
                peaklag(waveA,waveB,maxDelay,peakNums,minPeakSeparation);
            
            % Would need to fix this for something other than 50% duty
            % cycle !!
            if leading(1) == edges(1)
                peakIndsAbs{k,i,j} = peakIndsRel + edges((2*k-1)+(i-1)) - 2;
            else
                peakIndsAbs{k,i,j} = peakIndsRel + edges((2*k-1)+(2-i)) - 2;
            end
        end
        
        lagTime{i,j} = lagFrames{i,j}/sampleRate*1000; % [ms]
        wavespeed{i,j} = sum(travelDist(measPairs(j,1) : measPairs(j,2)-1))...
            ./lagTime{i,j};
        
        params.peaks.vals = peakVals;
        params.peaks.inds = peakIndsAbs;
    end
end

for j = 1 : size(measPairs,1)
    waveSpeed.push{1,j} = wavespeed{1,j}';
    if ~isempty(processedData.wave.release)
        waveSpeed.release{1,j} = wavespeed{2,j}';
    else
        waveSpeed.release = [];
    end
end
function [delay,pks,inds] = peaklag(x1,x2,maxdelay,peakNums,minSeparation)
% function [delay] = [delay,pks,inds] = peaklag(x1,x2,maxdelay,peakNums,minSeparation)
% find the delay between peaks in wave signals

x{1} = x1; x{2} = x2;
minH = 0.4*max(abs(x{1}));

% Finding peaks in first signal
[pks1,inds1] = findpeaks(abs(x{1}),'minpeakheight',minH);
% Retaining original sign
pks1 = pks1.*sign(x{1}(inds1));

% Keeping only peaks that aren't too close to each other; keeping first
% peak, unlike what 'minPeakDistance' does in 'findpeaks'
for n = 1 : 100;
    if min(diff(inds1)) < minSeparation
        for i = 1 : length(inds1) - 1
            if inds1(i+1) - inds1(i) < minSeparation
                if i+1 < length(inds1)
                    inds1 = [inds1(1:i);inds1(i+2:end)];
                    pks1 = [pks1(1:i);pks1(i+2:end)];
                    break
                else
                    inds1 = inds1(1:i);
                    pks1 = pks1(1:i);
                    break
                end
            end
        end
    else
        break
    end
end

% Retaining only those of interest (specified by user input)
if ~isempty(peakNums)
    % Throwing out peak specifications that are greater than the number of
    % peaks found
    peakNums = peakNums(peakNums <= length(pks1));
    
    if isempty(peakNums)
        disp(strvcat({'Specified peak(s) was/were not found',...
            'using first peak instead'}))
        peakNums = 1;
    end
    
    if length(pks1) >= length(peakNums) && length(pks1) >= max(peakNums)
        pks1 = pks1(peakNums); inds1 = inds1(peakNums);
    elseif length(pks1) <= max(peakNums)
        
    else
        pks1 = pks1(peakNums(1:length(pks1)));
        inds1 = inds1(peakNums(1:length(pks1)));
    end
end

% Finding corresponding peaks in second signal
for i = 1 : length(pks1)
    minH2 = abs(0.1*pks1(i));
    
    try [pks2(i,:),inds2(i,:)] = ...
            findpeaks(sign(pks1(i))*x{2}(inds1(i):inds1(i)+maxdelay),...
            'minpeakheight',minH2,'npeaks',1);
        inds2(i) = inds2(i) + inds1(i) - 1;
    catch
        try [pks2(i,:),inds2(i,:)] = findpeaks(sign(pks1(i))*x{2}(inds1(i):end),...
                'minpeakheight',minH2,'npeaks',1);
            inds2(i) = inds2(i) + inds1(i) - 1;
        catch
            pks2(i,:) = NaN; inds2(i,:) = NaN;
        end
    end
    
    if ~isnan(pks2(i,:))
        pks2(i,:) = pks2(i)*sign(x{2}(inds2(i)));
    end
end

% % Keeping only first peak 2 (taken care of by 'npeaks')
% pks2 = pks2(:,1);
% inds2 = inds2(:,1);

% Getting rid of duplicate values for peak 2
for i = 2 : length(inds2)
    if inds2(i) == inds2(i-1);
        inds2(i-1) = NaN;
        pks2(i-1) = NaN;
    end
end

% Keeping only peaks where pairs could be found between first and second
% signals
pks(:,1) = pks1(~isnan(inds2));
pks(:,2) = pks2(~isnan(inds2));
inds(:,1) = inds1(~isnan(inds2));
inds(:,2) = inds2(~isnan(inds2));



% Subsample interpolation to better approximate peak locations
for i = 1 : size(inds,1)
    for j = 1 : size(inds,2)
        peakSeg = abs(x{j}(inds(i,j)-1:inds(i,j)+1));
        wo = acos((peakSeg(1)+peakSeg(3))/(2*peakSeg(2)));
        theta = atan((peakSeg(1)-peakSeg(3))/(2*peakSeg(2)*sin(wo)));
        delta = -theta/wo;
        
        inds(i,j) = inds(i,j) + delta;
    end
end

% Finding delay between first and second signal
delays = inds(:,2) - inds(:,1);
delay = mean(delays);


function [waveSpeed,spectra] = wsCalcFreq(processedData,params,options)
% Calculates wave speed based on vibration frequency
% 
% Jack Martin - 06/02/17

wave{1} = processedData.wave.push;
if ~isempty(processedData.wave.release)
    wave{2} = processedData.wave.release;
end

if options.loadDataYesNo
    Load{1} = processedData.load.push;
    if ~isempty(processedData.load.release)
        Load{2} = processedData.load.release;
    end
end

L = options.tendonLength/1000;
sampleRate = params.sampleRate.waveData;
NFFT = 2^nextpow2(sampleRate);
range = 1 : 3000;
fre = sampleRate/2*linspace(0,1,NFFT/2+1);
fre = fre(range);

try
    windowInds = params.windowInds;
end

for i = 1 : length(wave) % push vs. release
    for j = 1 : size(wave{i},1) % over time
        for k = 1 : size(wave{i}{j},2) % which measurement location
            try options.window; 
                catch, windowInds = 1 : size(wave{i}{j},1); end
            wv = wave{i}{j}(windowInds,k);
            y{i,j,k} = fft(wv,NFFT)/length(wv);
            Y{i,j,k} = 2*abs(y{i,j,k}(range));
        end
        
        switch options.peakFindMethod
            case 'manual'
                for k = 1 : size(wave{i}{j},2) % which measurement location
                    YY(:,k) = Y{i,j,k};
                end
                
                [~,maxInd] = graphMinMax(YY,20,'max');
                
            case 'auto'
                for k = 1 : size(wave{i}{j},2) % which measurement location
                    [~,mInd(k)] = max(Y{i,j,k});
                end
                
                maxInd = round(mInd);
%                 maxInd = round(mean(mInd));
        end
        
        for k = 1 : size(wave{i}{j},2) % which measurement location
            freq(i,j,k) = fre(maxInd(k));
            wavespeed{i,k}(j,:) = 2*freq(i,j,k)*L;
        end
    end
end

if options.loadDataYesNo
    for j = 1 : size(wave{1},1)
        loadMean(j) = mean(Load{1}{j});
    end
    
    loadMean = (loadMean-min(loadMean))/(max(loadMean)-min(loadMean));
end

for k = 1 : size(wavespeed,2) % which measurement location
    waveSpeed.push{1,k} = wavespeed{1,k}; 
    if ~isempty(processedData.wave.release)
        waveSpeed.release{1,k} = wavespeed{2,k};
    else
        waveSpeed.release = [];
    end
end

spectra.frequency = fre;
for j = 1 : size(wave{1},1) % over time
    for k = 1 : size(wave{1}{j},2) % which measurement location
        spectra.power.push{k}(:,j) = Y{1,j,k};
    end
end

if ~isempty(processedData.wave.release)
    for j = 1 : size(wave{2},1) % over time
        for k = 1 : size(wave{2}{j},2) % which measurement location
            spectra.power.release{k}(:,j) = Y{2,j,k};
        end
    end
else
    spectra.power.release = [];
end
function [minOrMax,ind] = graphMinMax(vect,tolerance,minMax)
% Allows user to find local minimum or maximum based on graphical input.
% vector is the data in which min/max is to be found, and must be in the
% form of one or more columns
% tolerance is the number of points on either side of the user input
% location that will be searched
% minMax can be 'min' to search for minimum or 'max' to search for maximum
% 
% Jack Martin - 06/15/17

figure(740)
plot(vect)
title('Select a point close to min/max')
disp('Select a point close to min/max')
[xLoc,~] = ginput(1);

sectionStart = round(xLoc-tolerance);
if sectionStart < 1, sectionStart = 1; end

sectionEnd = round(xLoc+tolerance);
if sectionEnd > length(vect), sectionEnd = length(vect); end

section = sectionStart : sectionEnd;

for i = 1 : size(vect,2)
    switch minMax
        case 'min'
            [minOrMax(i),ind(i)] = min(vect(section,i));
        case 'max'
            [minOrMax(i),ind(i)] = max(vect(section,i));
    end
end

ind = ind + sectionStart - 1;

close(740)


function [waveSpeedOut] = wsProcessWaveSpeed(rawData,params,options)

ws{1} = rawData.waveSpeed.push;
if ~isempty(rawData.waveSpeed.release)
    ws{2} = rawData.waveSpeed.release;
end

if strcmp(options.waveSpeedMethod,'XCorr')
    waveCorr = params.waveCorrelation.rPeak;
    minCorr = options.minCorr;
end

deltaWSThresh = options.deltaWSThresh;
minSegLength = options.minSegLength;
filterLowPass = options.filterLowPass;
tapRate = params.tapTiming.tapRate;
switch options.waveSpeedMethod
    case 'frequency', measPairs = [1:size(ws{1},2)]';
    otherwise, measPairs = params.measPairs;
end

wsInf = 0;
wsLowCorr = 0;
wsExcessiveDelta = 0;
wsShortSegment = 0;
wsSize = 0;

for i = 1 : length(ws) % push vs. release
    for j = 1 : size(measPairs,1) % which measurement pair
        wsSize = wsSize + numel(ws{i}{j});
        wsDeleted{i}{j} = NaN(size(ws{i}{j}));
        
        % Get rid of Inf values
        ws{i}{j}(isinf(ws{i}{j})) = NaN;
        wsInf = wsInf + sum(isinf(ws{i}{j}));
        wsDeleted{i}{j}(isinf(ws{i}{j})) = 1;
        
        % Get rid of poor correlation points
        if strcmp(options.waveSpeedMethod,'XCorr')
            ws{i}{j}(waveCorr{i,j} < minCorr) = NaN;
            wsLowCorr = wsLowCorr + sum(waveCorr{i,j} < minCorr);
            wsDeleted{i}{j}(waveCorr{i,j} < minCorr) = 1;
        end
        
        % Throw out data where wave speed changes more than deltaWSThresh
        % from neighboring frames.
%         % Start search from min wave speed within trial b/c we trust these
%         % most (exclude start and end). Search in both directions.
%         wsLength = length(ws{i}{j});
%         wsSection = round(wsLength/10):round(9*wsLength/10);
%         [~,startInd] = min(ws{i}{j}(wsSection)); % could fix to go by diff(ws)
%         startInd = startInd + round(wsLength/10) - 1;
%         
%         % Search Forwards
%         for k = startInd : length(ws{i}{j}) - 1
%             
%         end
        for k = 2 : length(ws{i}{j}) - 1
            if abs(ws{i}{j}(k) - ws{i}{j}(k-1)) > deltaWSThresh && ...
                    abs(ws{i}{j}(k) - ws{i}{j}(k+1)) > deltaWSThresh
                wsNaN(i,j,k) = 1;
            else
                wsNaN(i,j,k) = 0;
            end
        end
        
        if abs(ws{i}{j}(1) - ws{i}{j}(2)) > deltaWSThresh,wsNaN(i,j,1) = 1;
        else,wsNaN(i,j,1) = 0;end
        
        if abs(ws{i}{j}(end) - ws{i}{j}(end-1)) > deltaWSThresh,wsNaN(i,j,k+1) = 1;
        else,wsNaN(i,j,k+1) = 0;end
        
        ws{i}{j}(wsNaN(i,j,:)==1) = NaN;
        wsExcessiveDelta = wsExcessiveDelta + sum(wsNaN(i,j,:)==1);
        wsDeleted{i}{j}(wsNaN(i,j,:)==1) = 1;
        
        % Throwing out short segments of data
        for k = 1 : length(ws{i}{j}) - minSegLength
            if sum(isnan(ws{i}{j}(k:k+minSegLength))) > 1
                wsNaN(i,j,k:k+minSegLength) = 1;
            end
        end
        
        ws{i}{j}(wsNaN(i,j,:)==1) = NaN;
        wsShortSegment = wsShortSegment + sum(wsNaN(i,j,:)==1);
        wsDeleted{i}{j}(wsNaN(i,j,:)==1) = 1;
        
        % Filling NaN gaps
        if options.nanFill
            wsNaNfill{i}{j} = inpaint_nans(ws{i}{j},4);
            % Filtering
            Wn = filterLowPass/(tapRate/2);
            [bf,af] = butter(2,Wn,'low');
            wsFilt{i}{j} = filtfilt(bf,af,wsNaNfill{i}{j});
        else
            % Filtering
            Wn = filterLowPass/(tapRate/2);
            [bf,af] = butter(2,Wn,'low');
            wsFilt{i}{j} = filtfilt(bf,af,ws{i}{j});
        end
    end
end

if wsInf > 0
    infWarning = [num2str(wsInf) '/' num2str(wsSize) ...
        ' Inf wave speed points have been deleted'];
    warning(infWarning)
end
if wsLowCorr > 0
    infWarning = [num2str(wsLowCorr) '/' num2str(wsSize) ...
        ' wave speed points have been deleted due to low wave correlation'];
    warning(infWarning)
end
if wsExcessiveDelta > 0
    deltaWarning = [num2str(wsExcessiveDelta) '/' num2str(wsSize) ...
        ' wave speed points have been deleted due to excessive rate of change (delta)'];
    warning(deltaWarning)
end
if wsShortSegment > 0
    segmentWarning = [num2str(wsShortSegment) '/' num2str(wsSize) ...
        ' wave speed points have been deleted due to short segment length'];
    warning(segmentWarning)
end

waveSpeedOut.filt.push = wsFilt{1};
waveSpeedOut.unfilt.push = ws{1};
waveSpeedOut.deleted.push = wsDeleted{1};
if ~isempty(rawData.waveSpeed.release)
    waveSpeedOut.filt.release = wsFilt{2};
    waveSpeedOut.unfilt.release = ws{2};
    waveSpeedOut.deleted.release = wsDeleted{2};
else
    waveSpeedOut.filt.release = [];
    waveSpeedOut.unfilt.release = [];
    waveSpeedOut.deleted.release = [];
end

% Define time vectors
sampleRateWave = params.sampleRate.waveData;
if options.tapDataYesNo
    leadingWaveCrop = params.tapTiming.leadingWaveCrop;
    trailingWaveCrop = params.tapTiming.trailingWaveCrop;
    indShift = round(mean(diff(params.tapTiming.edgesCrop))/2);
    pushTime = (leadingWaveCrop-1+indShift)/sampleRateWave*1000; % [ms]
    pushTime = pushTime(1:length(ws{1}{1}));
    
    if ~isempty(rawData.waveSpeed.release)
        releaseTime = (trailingWaveCrop-1+indShift)/sampleRateWave*1000;
        releaseTime = releaseTime(1:length(ws{2}{1}));
    end
else
    tapIndsWaveCrop = params.tapTiming.tapIndsWaveCrop;
    indShift = round(mean(diff(tapIndsWaveCrop))/2);
    pushTime = (tapIndsWaveCrop-1+indShift)/sampleRateWave*1000;
end

waveSpeedOut.time.push = pushTime;
if ~isempty(rawData.waveSpeed.release)
    waveSpeedOut.time.release = releaseTime;
else
    waveSpeedOut.time.release = [];
end

% Combined push and release (may want to redo -- combine before editing)
if ~isempty(rawData.waveSpeed.release)
    waveSpeedOut.filt.combined = [];
    waveSpeedOut.unfilt.combined = [];
    [waveSpeedOut.time.combined, indComb] = sort([pushTime; releaseTime]);
    
    for j = 1 : size(measPairs,1) % which measurement pair
        wsComb{j} = [ws{1}{j}; ws{2}{j}];
        wsComb{j} = wsComb{j}(indComb);
        wsFiltComb{j} = [wsFilt{1}{j}; wsFilt{2}{j}];
        wsFiltComb{j} = wsFiltComb{j}(indComb);
    end
    
    waveSpeedOut.filt.combined = wsFiltComb;
    waveSpeedOut.unfilt.combined = wsComb;
else
    waveSpeedOut.filt.combined = [];
    waveSpeedOut.unfilt.combined = [];
    waveSpeedOut.time.combined = [];
end


function [figHandles] = wsPlot(rawData,processedData,params,options)
% Plots results
%
% Jack Martin - 06/01/17

if ishandle(1001), close(1001), end
if ishandle(1002), close(1002), end

if options.plotYesNo
    % Determining which data were collected
    tyn = options.tapDataYesNo;
    lyn = options.loadDataYesNo;
    pyn = options.posDataYesNo;
    eyn = options.emgDataYesNo;
    numPlots = 2 + lyn + pyn + eyn;
    
    % Creating figure
    figNumber = options.figNumber;
    wsFig.fig = figure(figNumber);
    
    % Wave motion plot
    waveMot = processedData.wave.filtered;
    sampleRateWave = params.sampleRate.waveData;
    timeWave = processedData.wave.time; % [ms]
    try windowOption = options.window; catch, windowOption = []; end
    if tyn, tapSig = processedData.tapper.tapSig;
        tapTime = processedData.tapper.time;
        else, tapInds = params.tapTiming.tapIndsWaveCrop; end
    switch options.waveSpeedMethod
        case 'frequency', measPairs = 1:size(waveMot,2);
        otherwise, measPairs = params.measPairs; end
    
    % Creating subplot and defining options
    wsFig.axWave = subplot(numPlots,1,1);
    fsize1 = 14; fsize2 = 10;
    waveCol = [0 0 0; 0 0.7 0; 0.8 0 0; 0.6 0.6 0.6];
    tapCol = [0.6 0.8 1];
    xLims = [0 1.03*max(timeWave)];
    xLab = 'Time [ms]';
    switch options.collectionMethod
        case 'accelerometer', yLabWv = {'Transverse Acceleration' '[m/s^2]'};
        case { 'ultrasound', 'laser' }, yLabWv = {'Transverse Velocity' '[m/s]'}; end
    
    % Plotting boxes to indicate which data were used and plotting
    % tap signal/timing
    if ~isempty(windowOption)
        % A window was specified, so only a portion of wave motion data was
        % used
        windowInds = params.windowInds;
        
        maxWave = max(max(waveMot));
        minWave = min(min(waveMot));
        
        if tyn
            if ~isempty(processedData.waveSpeed.filt.release)
                startInds{1} = params.tapTiming.leadingWaveCrop;
                startInds{2} = params.tapTiming.trailingWaveCrop;
            else
                startInds{1} = params.tapTiming.leadingWaveCrop;
            end
            
            windowIndicator1 = zeros(size(waveMot,1),1);
            windowIndicator2 = zeros(size(waveMot,1),1);
            
            for i = 1 : length(startInds)
                for k = 1 : length(startInds{i}) - 1
                    if iscell(windowInds)
                        thisWindow = windowInds{i,1,k};
                    else
                        thisWindow = windowInds;
                    end
                    
                    windowIndicator1(startInds{i}(k)+thisWindow) = ...
                        1.2*max(max(waveMot(startInds{i}(k)+thisWindow-1,:)));
                    windowIndicator2(startInds{i}(k)+thisWindow) = ...
                        -1.2*max(max(-waveMot(startInds{i}(k)+thisWindow-1,:)));
                    
                    windowCol = [0.7 0.7 0.7];
                end
            end
            
            wsFig.window{1} = plot(timeWave,windowIndicator1,'color',windowCol);
            hold on
            wsFig.window{2} = plot(timeWave,windowIndicator2,'color',windowCol);
            
            % Plotting tap signal
            wsFig.tp = plot(timeWave,tapSig,'color',tapCol);
        else
            startInds = tapInds;
            for k = 1 : length(startInds) - 1
                if iscell(windowInds)
                    thisWindow = windowInds{k};
                else
                    thisWindow = windowInds;
                end
                
                wsFig.windows{k} = ...
                    plotWindowBox(thisWindow,timeWave,startInds(k),maxWave,minWave);
                hold on
            end
            
            % Plotting tap indices
            wsFig.tp = plot(timeWave(tapInds),zeros(1,length(tapInds)),...
                'marker','^','color',tapCol,'linestyle','none','linewidth',2);
        end
    else
        % No window was defined, so all data was used
        if tyn
            % Plotting tap signal
            wsFig.tp = plot(timeWave,tapSig,'color',tapCol);
        else
            % Plotting tap indices
            wsFig.tp = plot(timeWave(tapInds),zeros(1,length(tapInds)),...
                'marker','^','color',tapCol,'linestyle','none','linewidth',2);
        end
    end
    
    % Plotting correlations if using cross-correlation method
    if strcmp(options.waveSpeedMethod,'XCorr')
        lagFrames = params.waveCorrelation.lagFrames;
        for i = 1 : size(lagFrames,1)
            for j = 1 : size(lagFrames,2)
                for k = 1 : length(lagFrames{i,j})
                    if i == 1
                        wave = processedData.wave.push{k}(:,measPairs(j,1));
                        if tyn
                            startInd = params.tapTiming.leadingWaveCrop(k);
                        else
                            startInd = params.tapTiming.tapIndsWaveCrop(k);
                        end
                        
                    elseif i == 2
                        wave = processedData.wave.release{k}(:,measPairs(j,1));
                        
                        if tyn
                            startInd = params.tapTiming.trailingWaveCrop(k);
                        else
                            startInd = params.tapTiming.tapIndsWaveCrop(k);
                        end
                    end
                    
                    lag = lagFrames{i,j}(k);
                    
                    if iscell(windowInds)
                        thisWindow = windowInds{i,j,k};
                    else
                        thisWindow = windowInds;
                    end
                    
                    wsFig.shiftedWave{i,j,k} = ...
                        plot(timeWave(startInd+thisWindow-1+round(lag)),...
                        wave(thisWindow),':','color',waveCol(measPairs(j,1),:));
                    hold on
                end
            end
        end
        
        if options.plotCorr
            rVals = params.waveCorrelation.rVals;
            segLength = length(rVals{1,1,1});
            corrInds{1} = params.tapTiming.leadingWaveCrop;
            corrInds{2} = params.tapTiming.trailingWaveCrop;
            
            if ~isempty(windowOption)
                if isfloat(windowOption)
                    windowStart = params.windowInds(1);
                else
                    windowStart = 1;
                end
            else
                windowStart = 1;
            end
            
            yyaxis right
            for i = 1 : size(rVals,1)
                for j = 1 : size(rVals,2)
                    for k = 1 : size(rVals,3)
                        if isempty(rVals{i,j,k}),continue,end
                        wsFig.rVals{i,j,k} = ...
                            plot(timeWave(corrInds{i}(k)+windowStart-1: ...
                            corrInds{i}(k)+windowStart-1+segLength-1),...
                            rVals{i,j,k},'b-');
                        hold on
                    end
                end
            end
            ylabel('Correlation')
            ylim([-1 1])
            box off
            wsFig.axWave.YColor = [0 0 0];
            yyaxis left
        end
    end
    
    % Plotting peaks (if using peak-to-peak method
    if strcmp(options.waveSpeedMethod,'P2P')
        pks = params.peaks.vals;
        inds = params.peaks.inds;
        pkVect = [];
        indVect = [];
        
        for i = 1 : size(inds,2)
            for j = 1 : size(inds,3)
                for k = 1 : size(inds,1)
                    for m = 1 : size(inds{k,i,j},1)
                        for n = 1 : size(inds{k,i,j},2)
                            pkVect = [pkVect; pks{k,i,j}(m,n)];
                            indVect = [indVect; inds{k,i,j}(m,n)];
                        end
                    end
                end
            end
        end
        
        pkTime = indVect/sampleRateWave*1000;
        
        hold on
        plot(pkTime,pkVect,'b.','markersize',16)
    end
    
    % Plotting wave motion
    for i = 1 : size(waveMot,2)
        wsFig.wv{i} = plot(timeWave,waveMot(:,i),'-','color',waveCol(i,:));
        hold on
    end
    hold off
    xlim(xLims)
    xlabel(xLab)
    ylabel(yLabWv)
    box off
    
    % Wave speed plot
    sampleRateWS = params.tapTiming.tapRate;
    
    waveSpeedFilt{1} = processedData.waveSpeed.filt.push;
    waveSpeedUnfilt{1} = processedData.waveSpeed.unfilt.push;
    waveSpeedDeleted{1} = processedData.waveSpeed.deleted.push;
    
    timeWS{1} = processedData.waveSpeed.time.push;
    if ~isempty(processedData.waveSpeed.filt.release)
        
        waveSpeedFilt{2} = processedData.waveSpeed.filt.release;
        waveSpeedUnfilt{2} = processedData.waveSpeed.unfilt.release;
        waveSpeedDeleted{2} = processedData.waveSpeed.deleted.release;
        
        timeWS{2} = processedData.waveSpeed.time.release;
    end
    
    % Creating subplot and defining options
    wsFig.axWS = subplot(numPlots,1,2);
    wsCol = get(gca,'ColorOrder');
    wsDeletedCol = [1 0 0];
    lsFilt = [{'-'}, {':'}];
    lsRaw = [{'-'}, {'-.'}];
    deleteMarker = [{'*'}, {'o'}];
    yLabWS = {'Wave Speed' '[m/s]'};
    pushRelease = {'Push' 'Release'};
    switch options.waveSpeedMethod
        case 'frequency'
            for i = 1 : length(waveSpeedFilt)
                for j = 1 : length(waveSpeedFilt{i})
                    if length(waveSpeedFilt) == 1
                        legendWS{length(waveSpeedFilt{i})*(i-1)+j} = ...
                            ['Loc ' num2str(measPairs(j))];
                    else
                        legendWS{length(waveSpeedFilt{i})*(i-1)+j} = ...
                            ['Loc ' num2str(measPairs(j)) ' ' pushRelease{i}];
                    end
                end
            end
        otherwise
            for i = 1 : length(waveSpeedFilt)
                for j = 1 : length(waveSpeedFilt{i})
                    if length(waveSpeedFilt) == 1
                        legendWS{length(waveSpeedFilt{i})*(i-1)+j} = ...
                            [num2str(measPairs(j,1)) '-' num2str(measPairs(j,2))];
                    else
                        legendWS{length(waveSpeedFilt{i})*(i-1)+j} = ...
                            [num2str(measPairs(j,1)) '-' num2str(measPairs(j,2)) ' ' pushRelease{i}];
                    end
                end
            end
    end
    
    % Plotting wave speed
    for i = 1 : length(waveSpeedFilt)
        for j = 1 : length(waveSpeedFilt{i})
            wsFig.wsFilt{i,j} = plot(timeWS{i}(1:length(waveSpeedFilt{i}{j})),...
                waveSpeedFilt{i}{j},...
                'color',wsCol(j,:),'linestyle',lsFilt{i},'linewidth',2);
            hold on
        end
    end
    xlim(xLims)
    xlabel(xLab)
    ylabel(yLabWS)
    legend(legendWS,'location','best','orientation','horizontal')
    legend boxoff
    box off
    
    for i = 1 : length(waveSpeedUnfilt)
        for j = 1 : length(waveSpeedUnfilt{i})
            wsFig.wsUnfilt{i,j} = plot(timeWS{i}(1:length(waveSpeedUnfilt{i}{j})),...
                waveSpeedUnfilt{i}{j},...
                'color',wsCol(j,:),'linestyle',lsRaw{i},'linewidth',1);
            hold on
        end
    end
    
    % Plotting indicator for deleted wave speeds
    for i = 1 : length(waveSpeedFilt)
        for j = 1 : length(waveSpeedFilt{i})
            wsFig.wsDeleted{i,j} = plot(timeWS{i}(1:length(waveSpeedFilt{i}{j})),...
                waveSpeedDeleted{i}{j},...
                'color',wsDeletedCol,'linestyle',lsRaw{i},...
                'marker',deleteMarker{i},'linewidth',1,'markersize',4);
%             wsFig.wsDeleted{i,j} = plot(timeWS{i}(1:length(waveSpeedFilt{i}{j})),...
%                 waveSpeedDeleted{i}{j},...
%                 'color',wsDeletedCol,'linestyle',lsFilt{i},...
%                 'linewidth',2,'markersize',4);
            hold on
        end
    end
    hold off
    
    % Load plot
    n = 3;
    if lyn
        try sampleRate = rawData.lvm.sampleRate;
            catch, sampleRate = rawData.mat.sampleRate; end
        Load = processedData.load.filtered;
        loadTime = processedData.load.time; % [ms]
        
        % Creating subplot and defining options
        wsFig.axLoad = subplot(numPlots,1,3);
        loadCol = [0.6 0 0];
        yLabLoad = 'Load';
        
        % Plotting load data
        wsFig.ld = plot(loadTime,Load,'color',loadCol,'linewidth',2);
        xlim(xLims)
        xlabel(xLab)
        ylabel(yLabLoad)
        box off
        n = n + 1;
    end
    
    % Position plot
    if pyn
        try sampleRate = rawData.lvm.sampleRate;
            catch, sampleRate = rawData.mat.sampleRate; end
        pos = processedData.pos.filtered;
        posTime = processedData.pos.time; % [ms]
        
        % Creating subplot and defining options
        wsFig.axPos = subplot(numPlots,1,n);
        posCol = [0 0 0.6];
        yLabPos = 'Position';
        
        % Plotting position data
        wsFig.ps = plot(posTime,pos,'color',posCol,'linewidth',2);
        xlim(xLims)
        xlabel(xLab)
        ylabel(yLabPos)
        box off
        n = n + 1;
    end
    
    % EMG plot
    if eyn
        try sampleRate = rawData.lvm.sampleRate;
            catch, sampleRate = rawData.mat.sampleRate; end
        emg = processedData.emg.rectNorm;
        emgTime = processedData.emg.time; % [ms]
        
        % Creating subplot and defining options
        wsFig.axEmg = subplot(numPlots,1,n);
        emgCol = flipud(get(gca,'ColorOrder'));
        yLabEMG = 'Muscle Activity';
        
        % Plotting EMG data
        for i = 1 : size(emg,2)
            wsFig.em{i} = plot(emgTime,emg(:,i),'color',emgCol(i,:));
            hold on
        end
        hold off
        xlim(xLims)
        xlabel(xLab)
        ylabel(yLabEMG)
        box off
    end
    
    % Figure options
    set(wsFig.fig,'color',[1 1 1])
    
    % Linking axes
    if ~lyn && ~pyn && ~eyn
        linkaxes([wsFig.axWave wsFig.axWS],'x')
    elseif lyn && ~pyn && ~eyn
        linkaxes([wsFig.axWave wsFig.axWS wsFig.axLoad],'x')
    elseif ~lyn && pyn && ~eyn
        linkaxes([wsFig.axWave wsFig.axWS wsFig.axPos],'x')
    elseif ~lyn && ~pyn && eyn
        linkaxes([wsFig.axWave wsFig.axWS wsFig.axEmg],'x')
    elseif lyn && pyn && ~eyn
        linkaxes([wsFig.axWave wsFig.axWS wsFig.axLoad wsFig.axPos],'x')
    elseif lyn && ~pyn && eyn
        linkaxes([wsFig.axWave wsFig.axWS wsFig.axLoad wsFig.axEmg],'x')
    elseif ~lyn && pyn && eyn
        linkaxes([wsFig.axWave wsFig.axWS wsFig.axPos wsFig.axEmg],'x')
    elseif lyn && pyn && eyn
        linkaxes([wsFig.axWave wsFig.axWS wsFig.axLoad wsFig.axPos wsFig.axEmg],'x')
    end
    
    % FFT plot
    switch options.waveSpeedMethod
        case 'frequency'
            fre = processedData.spectra.frequency;
            spectra{1} = processedData.spectra.power.push;
            if ~isempty(processedData.spectra.power.release)
                spectra{2} = processedData.spectra.power.release;
            end
            
            if lyn
                Loads{1} = processedData.load.push;
                if ~isempty(processedData.spectra.power.release)
                    Loads{2} = processedData.load.release;
                end
                
                for i = 1 : length(Loads)
                    for k = 1 : length(Loads{i})
                        loadMean(i,k) = mean(Loads{i}{k});
                    end
                    
                    loadMean(i,:) = (loadMean(i,:)-min(loadMean(i,:)))...
                        /(max(loadMean(i,:))-min(loadMean(i,:)));
                end
            end
            
            % Creating figure and defining options
            wsFig.fig2 = figure(figNumber+1);
            locCol = [1 0 0; 0.8 0.8 0; 0 0.7 0; 0 0.8 0.8; 0 0 1];
            
            % Creating subplots for push vs. release and locations
            for i = 1 : length(spectra)
%                 axSpectra(i) = subplot(length(spectra),1,i);
                for j = 1 : length(spectra{i})
                    axSpectra(i,j) = ...
                        subplot(length(spectra),length(spectra{i}),...
                        length(spectra{i})*(i-1)+j);
                    
                    if length(spectra) == 1
                        spectraTitle{length(spectra{i})*(i-1)+j} = ...
                            ['Loc ' num2str(measPairs(j))];
                    else
                        spectraTitle{length(spectra{i})*(i-1)+j} = ...
                            ['Loc ' num2str(measPairs(j)) ' ' pushRelease{i}];
                    end
                    
                    for k = 1 : size(spectra{i}{j},2)
                        if lyn
                            plot(fre,spectra{i}{j}(:,k),...
                                'color',locCol(j,:)*loadMean(i,k))
                        else
                            plot(fre,spectra{i}{j}(:,k),...
                                'color',locCol(j,:))
                        end
                        hold on
                    end
                    
                    hold off
                    box off
                    xlim([0 options.filterBandWave(2)])
                    xlabel('Frequency [Hz]')
                    ylabel('Power')
                    title(spectraTitle{length(waveSpeedFilt{i})*(i-1)+j})
                end
            end
            
            set(wsFig.fig2,'color',[1 1 1])
            linkaxes(axSpectra)
    end
    
    % Defining output
    figure(figNumber)
    figHandles = wsFig;
else
    figHandles = [];
end

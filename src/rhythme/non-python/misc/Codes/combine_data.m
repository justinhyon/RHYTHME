path = uigetdir(pwd, 'Select a folder');
files = dir(fullfile(path, '*.set'));

% Get trial names
concatnames = [];
for i = 1:size(files, 1)
    if contains(string(files(i).name), "trim")
        concatnames = [concatnames extractBefore(string(files(i).name),"_trim")];
    end
end
concatnames = unique(concatnames);

% Concatenate files per trial, assumes that there are only two files per
% trial
for i = 1:size(concatnames, 2)
    clear file1 file2 ALLEEG EEG CURRENTSET ALLCOM
    [file1, file2] = files(find(contains({files.name},concatnames(i)))).name;
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    % [file1, ~] = uigetfile('*.set');
    % [file2, path] = uigetfile('*.set');
    EEG = pop_loadset('filename',{file1 file2},'filepath',path);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'study',0); 

    EEG.data = [ALLEEG(1).data; ALLEEG(2).data];
    EEG.nbchan = 256;
    EEG.chanlocs = [ALLEEG(1).chanlocs ALLEEG(2).chanlocs];
    
    % Create stim channel
    eventlocs = zeros(size(EEG.event, 2), 1);
    for j = 1:size(EEG.event, 2)
        n = ALLEEG(2).event(j).latency;
        [minValue,closestIndex] = min(abs(EEG.times - n));
        eventlocs(j) = closestIndex;
    end
    % eventlocs = unique(eventlocs);
    
    stim = zeros(1, size(EEG.times, 2));
    eventtypes = {EEG.event.type};
    stim(eventlocs) = str2double(eventtypes);
    
    EEG.data = [EEG.data; stim];

    pop_saveset(EEG, char(concatnames(i)), path)
end

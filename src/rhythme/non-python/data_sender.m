% addpath(genpath("/Users/justinhyon/Downloads/fieldtrip-20220104"));
currentFile = mfilename( 'fullpath' );
[pathstr,~,~] = fileparts( currentFile );
addpath( fullfile( pathstr, 'data' ) );
cfg.source.dataset = 'data/Testdata20240102-3.bdf'; %The source of the data is configured as
%cfg.channel        = 1:257;                         % list with channel "names"
%cfg.maxblocksize   = 5;                             % seconds
cfg.speed          = 1;                             % Seconds
%cfg.readevent      = 'yes';
%cfg.fsample        = 2048;                         % sampling frequency
cfg.target.dataset = 'buffer://localhost:1972';     % where to write the data
ft_realtime_fileproxy(cfg)
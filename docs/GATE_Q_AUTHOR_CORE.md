# Gate Q author core trial construction

## `TSS-main/tss/utils/get_subj_eeg.m`

```matlab
0001: function dat = get_subj_eeg(subjid, csc_dir, img_dir, read_ep, ...
0002:     read_rest, read_cl, read_cl_control, phase_chan, dummy_flag)
0003: %GET_SUBJ_EEG loads eeg for a subject from disk
0004: %
0005: % Input:
0006: %   subjid: subject ID
0007: %   csc_dir: directory containing csc data
0008: %   img_dir: directory containing img data
0009: %   read_ep: whether to read in EP data
0010: %   read_rest: whether to read in rest data
0011: %   read_cl: whether to read in closed-loop data
0012: %   read_cl_control: whether to read in closed-loop control data
0013: %   phase_chan: whether to read in phase channel data
0014: %   dummy_flag: whether to read in dummy events
0015: %
0016: % Output:
0017: %   dat: structure containing epoched data for different conditions
0018: %
0019: 
0020: if ~exist('dummy_flag','var')
0021:     dummy_flag = false; %whether or not to read in dummy events from first channel on box D
0022: end
0023: 
0024: % read jackbox information for translating CSC numbers to correct contact
0025: % labels
0026: jacksheet = readtable(fullfile(img_dir, 'jackbox.csv'));
0027: contacts = jacksheet.Elect;
0028: 
0029: % find the contacts that we want to include, based on stimulation
0030: % information
0031: 
0032: stim = get_stim_info(subjid);
0033: 
0034: if ~strcmp(subjid,'p20')
0035: chan_include = startsWith(contacts, stim.anode(isletter(stim.anode))) | ...
0036:     startsWith(contacts, stim.cathode(isletter(stim.cathode))) | ... 
0037:     startsWith(contacts, stim.phase(isletter(stim.phase))) | ...
0038:     startsWith(contacts, 'TTL16');
0039: else
0040:     labels = {stim.anode{:} stim.cathode{:}};
0041:     
0042:     chan_include = false(size(contacts));
0043:     for l = 1:length(labels)
0044:         chan_include = chan_include | startsWith(contacts, labels{l});
0045:     end    
0046:     
0047: end
0048: 
0049: if exist('phase_chan','var')
0050:     if strcmp(phase_chan,'all')
0051:         chan_include = ~strcmp(contacts,'') & ...
0052:         ~startsWith(contacts, 'EKG') & ...
0053:         ~startsWith(contacts, 'REF') & ...
0054:         ~startsWith(contacts, 's');
0055:     else        
0056:         if phase_chan
0057:             phase_chan = get_phase_chan(subjid, contacts);
0058:             chan_include  = chan_include & ismember(contacts, phase_chan);
0059:         end
0060:     end
0061: end
0062: 
0063: if strcmp(subjid,'UC004')
0064:     chan_include(1) = false;
0065: end
0066: 
0067: % durations to epoch around EP events
0068: ep_pre_dur = 500;
0069: ep_post_dur = 1500;
0070: 
0071: % durations to epoch around rest
0072: 
0073: if ~dummy_flag
0074:     rest_pre_dur =  -15000; % 5sec after, -5000
0075:     rest_post_dur = 119000; % two minutes
0076: else
0077:     rest_pre_dur =  0; % 10sec after
0078:     rest_post_dur = 590000;
0079: end
0080: 
0081: % durations to epoch around closed-loop stim
0082: cl_pre_dur = 1000;
0083: cl_post_dur = 2000;
0084: 
0085: % timestamps for different events vary by subject, reading from a file that
0086: % contains this information
0087: 
0088: timestamps = get_condition_info(subjid);
0089: dat = struct; % init to empty
0090: if read_ep
0091:     prefix = 'ep'; % pre/post closed-loop
0092:     if isfield(timestamps, prefix)
0093:         dat = get_condition_dat(prefix, ep_pre_dur, ep_post_dur, dat);
0094:     end
0095:     
0096:     prefix = 'ep_control'; % blind
0097:     if isfield(timestamps, prefix)
0098:         dat = get_condition_dat(prefix, ep_pre_dur, ep_post_dur, dat);
0099:     end
0100:     
0101:     % added for p20
0102:     prefix = 'ep_other_ltc'; % blind
0103:     if isfield(timestamps, prefix)
0104:         dat = get_condition_dat(prefix, ep_pre_dur, ep_post_dur, dat);
0105:     end
0106:     
0107:     prefix = 'ep_other_hc'; % blind
0108:     if isfield(timestamps, prefix)
0109:         dat = get_condition_dat(prefix, ep_pre_dur, ep_post_dur, dat);
0110:     end
0111:     
0112:     prefix = 'ep_control_ltc'; % blind
0113:     if isfield(timestamps, prefix)
0114:         dat = get_condition_dat(prefix, ep_pre_dur, ep_post_dur, dat);
0115:     end
0116:     
0117:     prefix = 'ep_control_hc'; % blind
0118:     if isfield(timestamps, prefix)
0119:         dat = get_condition_dat(prefix, ep_pre_dur, ep_post_dur, dat);
0120:     end
0121:  
0122: end
0123: 
0124: if read_rest
0125:     prefix = 'rest'; % pre/post closed-loop
0126:     if isfield(timestamps, prefix)
0127:         dat = get_condition_dat(prefix, rest_pre_dur, rest_post_dur, dat);
0128:     end
0129:     
0130:     prefix = 'rest_control'; % pre/post sham
0131:     if isfield(timestamps, prefix)
0132:         dat = get_condition_dat(prefix, rest_pre_dur, rest_post_dur, dat);
0133:     end
0134: end
0135: 
0136: if read_cl % read in closed-loop period data
0137:     prefix = 'cl'; % closed-loop
0138:     if isfield(timestamps, prefix)
0139:         dat = get_condition_dat(prefix, cl_pre_dur, cl_post_dur, dat);
0140:     end
0141: end
0142: 
0143: if read_cl_control % read in closed-loop period data
0144:     prefix = 'cl_control'; % closed-loop
0145:     if isfield(timestamps, prefix)
0146:         dat = get_condition_dat(prefix, cl_pre_dur, cl_post_dur, dat);
0147:     end
0148: end
0149: 
0150: % reorder
0151: dat = orderfields(dat);
0152: 
0153:     function dat = get_condition_dat(prefix, pre_dur, post_dur, dat)
0154:         
0155:         conds = fieldnames(timestamps.(prefix));
0156:         
0157:         for c = 1:length(conds)
0158:             dat = get_dat(conds{c}, prefix, dat, pre_dur, post_dur);
0159:         end
0160:         
0161:     end
0162: 
0163:     function dat = get_dat(cond, prefix, dat, pre_dur, post_dur)
0164:         
0165:         if ~strcmp(prefix, 'cl')
0166:             dat_name = [prefix '_' cond(1:end-4) '_epoched'];
0167:         else
0168:             dat_name = [prefix '_' cond(2:end-4) '_epoched'];
0169:         end
0170:         
0171:         if ~strcmp(prefix, 'cl')
0172:             time_name = [prefix '_' cond(1:end-4) '_time'];
0173:         else
0174:             time_name = [prefix '_' cond(2:end-4) '_time'];
0175:         end
0176:         
0177:         if isequal(subjid, 'p09') 
0178:             zeros_flag = true; % data in separate files pre and post for first 64 channels
0179:         else
0180:             zeros_flag = false;
0181:         end
0182:         
0183:         % load channel data
0184:         [dat.(dat_name), dat.contacts, dat.csc_names, dat.sr, dat.(time_name)] = get_epoched_eeg(csc_dir, timestamps.(prefix).(cond), ...
0185:             pre_dur, post_dur, contacts, chan_include, zeros_flag, dummy_flag);
0186:         
0187:     end
0188: 
0189: end
```

## `TSS-main/tss/preprocessing/run_cl_preproc.m`

```matlab
0001: function dat = run_cl_preproc(dat, fs, contacts)
0002: % RUN_CL_PREPROC runs preprocessing for closed-loop data
0003: %
0004: % Input:
0005: %   dat: continuous data structure
0006: %   fs: sampling rate
0007: %   contacts: cell array of contact names
0008: %
0009: % Output:
0010: %   dat: continuous data structure compatible with FieldTrip
0011: %
0012: 
0013: dat = epoched2dat(dat, contacts, fs);
0014: 
0015: if length(dat.label) > 1 % bipolar
0016:     cfg = [];
0017:     cfg.channel = 'all'; 
0018:     cfg.reref = 'no';
0019:     
0020:     montage.labelold = dat.label;
0021:     montage.labelnew = {[dat.label{1} '-' dat.label{2}]};
0022:     montage.tra = [1 -1];
0023:     
0024:     cfg.montage = montage;
0025:     
0026:     dat = ft_preprocessing(cfg, dat);
0027: end
0028: 
0029: dat = run_preproc(dat);
0030: 
0031: % correct timing
0032: for t = 1:length(dat.time)
0033:     dat.time{t} = dat.time{t} - 1;
0034: end
0035: 
0036: end
0037: 
0038: 
0039: function datout = run_preproc(datin)
0040: % RUN_PREPROC runs preprocessing for the input data
0041: %
0042: % Input:
0043: %   datin: continuous data structure
0044: %
0045: % Output:
0046: %   datout: continuous data structure with preprocessing applied
0047: %
0048: 
0049: cfg = [];
0050: cfg.latency = [datin.time{1}(1) datin.time{1}(end-1)]; %samples for dft
0051: datout = ft_selectdata(cfg, datin);
0052: 
0053: cfg = [];
0054: cfg.dftfilter = 'yes';
0055: cfg.dftfreq = [60 120 180];
0056: cfg.dftbandwidth = [.5 .5 .5];
0057: cfg.dftneighbourwidth = [1.5 1.5 1.5];
0058: cfg.dftreplace = 'neighbour';
0059: 
0060: datout = ft_preprocessing(cfg, datout);
0061: 
0062: cfg = [];
0063: cfg.latency = [-1 1];
0064: datout = ft_selectdata(cfg, datout);
0065: 
0066: end
0067: 
0068: function dat = epoched2dat(epoched_dat, contacts, sr)
0069: % EPOCHED2DAT converts epoched data to continuous data format
0070: %
0071: % Input:
0072: %   epoched_dat: epoched data matrix (channels x epochs x time points)
0073: %   contacts: cell array of contact names
0074: %   sr: sampling rate
0075: %
0076: % Output:
0077: %   dat: continuous data structure compatible with FieldTrip
0078: 
0079: dat.dimord = 'chan_rpt_time';
0080: dat.label = contacts;
0081: 
0082: for i = 1:size(epoched_dat,2)
0083:     if length(contacts)==1
0084:     dat.trial{i} = squeeze(epoched_dat(:,i,:))';
0085:     else
0086:     dat.trial{i} = squeeze(epoched_dat(:,i,:));    
0087:     end
0088:     dat.time{i} = linspace(0, size(epoched_dat,3)/sr, size(epoched_dat,3));
0089: end
0090: 
0091: end
```

## `TSS-main/tss/preprocessing/run_line_filter.m`

```matlab
0001: function datout = run_line_filter(datin, contacts, sr)
0002: % RUN_LINE_FILTER runs line noise filtering for resting state data
0003: %
0004: % Input:
0005: %   datin: continuous data structure
0006: %   contacts: contact names
0007: %   sr: sampling rate
0008: %
0009: % Output:
0010: %   datout: continuous data structure with line noise removed
0011: %
0012: 
0013: datin = epoched2dat(datin, contacts, sr);
0014: 
0015: cfg = [];
0016: cfg.bsfilter = 'yes';
0017: cfg.bsfreq = [59 61; 119 121; 179 181];
0018: 
0019: datout = ft_preprocessing(cfg, datin);
0020: 
0021: end
0022: 
0023: function dat = epoched2dat(epoched_dat, contacts, sr)
0024: % EPOCHED2DAT converts epoched data to continuous data format
0025: %
0026: % Input:
0027: %   epoched_dat: epoched data matrix (channels x epochs x time points)
0028: %   contacts: cell array of contact names
0029: %   sr: sampling rate
0030: %
0031: % Output:
0032: %   dat: continuous data structure compatible with FieldTrip
0033: 
0034: dat.dimord = 'chan_rpt_time';
0035: dat.label = contacts;
0036: 
0037: for i = 1:size(epoched_dat,2)
0038:     if length(contacts)==1
0039:     dat.trial{i} = squeeze(epoched_dat(:,i,:))';
0040:     else
0041:     dat.trial{i} = squeeze(epoched_dat(:,i,:));    
0042:     end
0043:     dat.time{i} = linspace(0, size(epoched_dat,3)/sr, size(epoched_dat,3));
0044: end
0045: 
0046: end
```

## `TSS-main/tss/utils/get_phase_chan.m`

```matlab
0001: function phase_chan = get_phase_chan(subjid, contacts)
0002: % GET_PHASE_CHAN returns phase channel for a given subject
0003: % 
0004: % Input:
0005: %   subjid: subject ID
0006: %   contacts: contact names
0007: %
0008: % Output:
0009: %   phase_chan: phase channel name
0010: %
0011: 
0012: % stim and phase lock info
0013: stim = get_stim_info(subjid);
0014: 
0015: if contains(stim.phase,'-')
0016:     stim.phase = strsplit(stim.phase,'-');
0017: end
0018: 
0019: is_phase = false(size(contacts));
0020: for c = 1:length(contacts)
0021:     if any(strcmp(contacts{c}, stim.phase))
0022:         is_phase(c) = true;
0023:     end
0024: end
0025: if ~any(is_phase)
0026:     is_phase(1) = true; % medial most
0027: end
0028: 
0029: phase_chan = contacts(is_phase);
0030: 
0031: end
```

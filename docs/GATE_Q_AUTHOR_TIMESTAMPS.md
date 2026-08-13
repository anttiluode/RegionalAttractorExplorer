# Gate Q author timestamp/event logic

## `TSS-main/tss/utils/get_condition_info.m`

```matlab
0001: function timestamps = get_condition_info(subjid)
0002: %GET_CONDITION_INFO returns timestamps of TTLs (index of rising edge) to be
0003: %used for extracting data
0004: %
0005: % Input:
0006: %   subjid: subject ID
0007: %
0008: % Output:
0009: %   timestamps: structure containing timestamps for EPs, rest,
0010: %               and closed-loop stim
0011: %
0012: 
0013: switch subjid
0014:     
0015:     case 'p09'
0016: 
0017:         % EPs - no control condition
0018:         timestamps.ep.pre_idx = 63:154;
0019:         timestamps.ep.post_idx = 1237:1296;
0020: 
0021:         % where does rest start before and after closed loop (nearest
0022:         % event)
0023:         timestamps.rest.pre_idx = 154;
0024:         timestamps.rest.post_idx = 1236;
0025: 
0026:         % timestamps for closed-loop stim
0027:         timestamps.cl.idx = 155:1236;
0028:         
0029:     case 'p11'
0030:         
0031:          % EPs - pre/post for trough and control, as well as longer duration
0032:         % post closed-loop
0033:         
0034:         timestamps.ep.pre_idx = 167:227;
0035:         timestamps.ep.post_idx = 1431:1502;
0036:         
0037:         timestamps.ep_control.pre_idx = 1503:1568;
0038:         timestamps.ep_control.post_idx = 2719:2792;
0039:         
0040:         % where does rest start before and after closed loop (nearest
0041:         % event)
0042:         timestamps.rest.pre_idx = 227;
0043:         timestamps.rest.post_idx = 1430; 
0044:         
0045:         timestamps.rest_control.pre_idx = 1568;
0046:         timestamps.rest_control.post_idx = 2718; 
0047:         
0048:         % timestamps for closed-loop stim
0049:         timestamps.cl.idx = 228:1430;
0050:         timestamps.cl_control.idx = 1569:2718;
0051:         
0052:     case 'p15'
0053:     
0054:         % EPs - no control condition
0055:         timestamps.ep.pre_idx = 1:59;
0056:         timestamps.ep.post_idx = 1151:1214;
0057: 
0058:         % where does rest start before and after closed loop (nearest
0059:         % event)
0060:         timestamps.rest.pre_idx = 59;
0061:         timestamps.rest.post_idx = 1150; % this period looks short in duration
0062: 
0063:         % timestamps for closed-loop stim
0064:         timestamps.cl.idx = 60:1150;        
0065:         
0066:     case 'p16'
0067:         
0068:         % EPs - no control condition
0069:         timestamps.ep.pre_idx = 1:90;
0070:         timestamps.ep.post_idx = 1147:1206;
0071:         
0072:         % where does rest start before and after closed loop (nearest
0073:         % event)
0074:         timestamps.rest.pre_idx = 90;
0075:         timestamps.rest.post_idx = 1146; % this period looks short in duration
0076:         
0077:         % timestamps for closed-loop stim
0078:         timestamps.cl.idx = 91:1146;
0079:         
0080:     case 'p17'
0081:         
0082:         % EPs - pre/post for trough and control, as well as longer duration
0083:         % post closed-loop
0084:         
0085:         timestamps.ep.pre_idx = 69:131;
0086:         timestamps.ep.post_idx = 1139:1197;
0087:         timestamps.ep.post_long_idx = 1198:1257;
0088:         
0089:         timestamps.ep_control.pre_idx = 1258:1317;
0090:         timestamps.ep_control.post_idx = 2460:2519;
0091:         
0092:         % where does rest start before and after closed loop (nearest
0093:         % event)
0094:         timestamps.rest.pre_idx = 131;
0095:         timestamps.rest.post_idx = 1197;
0096:         
0097:         timestamps.rest_control.pre_idx = 1257;
0098:         timestamps.rest_control.post_idx = 2519; 
0099:         
0100:         % timestamps for closed-loop stim
0101:         timestamps.cl.idx = 132:1138;
0102:         timestamps.cl_control.idx = 1318:2459;
0103:         
0104:     case 'p18'
0105:             
0106:         % EPs - no control condition
0107:         timestamps.ep.pre_idx = 53:112;
0108:         timestamps.ep.post_idx = 975:1034;
0109:         
0110:         % where does rest start before and after closed loop (nearest
0111:         % event)
0112:         timestamps.rest.pre_idx = 112;
0113:         timestamps.rest.post_idx = 974; % this period looks short in duration
0114:         
0115:         % timestamps for closed-loop stim
0116:         timestamps.cl.idx = 113:974;
0117:         
0118:     case 'p19'
0119:         
0120:         % EPs - pre/post for trough and control
0121:         
0122:         timestamps.ep_control.pre_idx = 2:61;
0123:         timestamps.ep_control.post_idx = 651:710;
0124:         
0125:         timestamps.ep.pre_idx = 711:770;
0126:         timestamps.ep.post_idx = 1294:1353;
0127:         
0128:         % where does rest start before and after closed loop (nearest
0129:         % event)
0130:         timestamps.rest_control.pre_idx = 61;
0131:         timestamps.rest_control.post_idx = 650;
0132:         
0133:         timestamps.rest.pre_idx = 770;
0134:         timestamps.rest.post_idx = 1293;
0135:         
0136:         % timestamps for closed-loop stim
0137:         timestamps.cl_control.idx = 62:650;
0138:         timestamps.cl.idx = 771:1293;
0139:         
0140:     case 'p20'
0141:         
0142:         % bilateral mapping
0143:         timestamps.ep_other_ltc.idx = 1:143;
0144:         timestamps.ep_other_hc.idx = 150:252;
0145:         
0146:         % pre and post eps
0147:         timestamps.ep_control_ltc.pre_idx = 253:403;
0148:         timestamps.ep_control_ltc.post_idx = 1680:1828;
0149:         timestamps.ep_control_hc.pre_idx = 404:505;
0150:         timestamps.ep_control_hc.post_idx = 1832:1945;
0151:             
0152:         % rest times (2 min)
0153:         timestamps.rest_control.pre_idx = 505;
0154:         timestamps.rest_control.post_idx = 1679;
0155:         
0156:         % timestamps for closed-loop stim
0157:         timestamps.cl_control.idx = 509:1679;
0158: 
0159:     case 'UC004'
0160: 
0161:         % EPs - pre/post for control
0162:         timestamps.ep_control.pre_idx = 77:116;
0163:         timestamps.ep_control.post_idx = 1318:1357;
0164: 
0165:         % where does rest start before and after closed loop (nearest event
0166:         % before)
0167:         timestamps.rest_control.pre_idx = 116;
0168:         timestamps.rest_control.post_idx = 1317;
0169: 
0170:         %timestamps for stim
0171:         timestamps.cl_control.idx = 117:1317;   
0172: 
0173:     case 'UC005'
0174: 
0175:         % EPs - pre/post for control
0176:         timestamps.ep_control.pre_idx = 39:78;
0177:         timestamps.ep_control.post_idx = 1279:1318;
0178: 
0179:         % where does rest start before and after closed loop (nearest event
0180:         % before)
0181:         timestamps.rest_control.pre_idx = 78;
0182:         timestamps.rest_control.post_idx = 1278;
0183: 
0184:         %timestamps for stim
0185:         timestamps.cl_control.idx = 79:1277;
0186: 
0187: end
0188: 
```

## `TSS-main/tss/utils/get_epoched_eeg.m`

```matlab
0001: function [epoched_dat, contacts, csc_names, sr, time] = get_epoched_eeg(csc_dir, timestamps, ...
0002:     pre_dur, post_dur, contacts, chan_idx, zeros_flag, dummy_flag)
0003: %GET_EPOCHED_EEG extracts epoched EEG data from Neuralynx CSC files.
0004: %
0005: % Input:
0006: %   csc_dir: directory containing CSC files
0007: %   timestamps: timestamps for events
0008: %   pre_dur: duration before event (ms)
0009: %   post_dur: duration after event (ms)
0010: %   contacts: contact names
0011: %   chan_idx: channel index
0012: %   zeros_flag: flag to remove zero CSC files
0013: %   dummy_flag: flag to include dummy TTL channel
0014: %
0015: % Output:
0016: %   epoched_dat: epoched EEG data
0017: %   contacts: contact names
0018: %   csc_names: CSC file names
0019: %   sr: sampling rate
0020: %   time: time vector
0021: %
0022: 
0023: if ~exist('chan_idx','var')
0024:     chan_idx = true(size(contacts));
0025: end
0026: 
0027: csc_files = dir([csc_dir filesep '*.ncs']);
0028: 
0029: if zeros_flag 
0030:     to_remove = false(size(csc_files));
0031:     for c = 1:length(csc_files)
0032:         if ~contains(csc_files(c).name, '0001') && ...
0033:                 exist([csc_files(c).folder filesep ...
0034:                        csc_files(c).name(1:end-4) '_0001.ncs'], 'file')
0035:             to_remove(c) = true;
0036:         end
0037:     end
0038:     csc_files(to_remove) = [];
0039: end
0040: 
0041: % resort csc_files by number, not alphabetically
0042: cno = nan(1, length(csc_files));
0043: for i = 1:length(csc_files)
0044:     cno(i) = sscanf(csc_files(i).name,'CSC%d');
0045: end
0046: [~, i] = sort(cno);
0047: csc_files = csc_files(i);
0048: csc_names = {csc_files(:).name}';
0049: 
0050: % Adjust the channel index to match the number of available CSC files,
0051: % as not all channels are input into the Neuralynx system
0052: 
0053: tmp_idx = 1:length(csc_files);
0054: if length(tmp_idx) > length(contacts)
0055:     tmp_idx = tmp_idx(1:length(contacts));
0056: end
0057: 
0058: chan_idx = chan_idx(tmp_idx);
0059: contacts = contacts(tmp_idx);
0060: 
0061: if dummy_flag
0062:     chan_idx(193) = true;
0063: end
0064: 
0065: % include only channels of interest
0066: csc_names = csc_names(chan_idx);
0067: csc_files = csc_files(chan_idx);
0068: if length(contacts) < find(chan_idx, 1, 'last' )
0069:     contacts{find(chan_idx, 1, 'last' )} = 'TTL';
0070: end
0071: contacts = contacts(chan_idx);
0072: 
0073: if dummy_flag
0074:     contacts{end} = 'TTL';
0075: end
0076: 
0077: % exclude empty or scalp channels
0078: to_remove = false(size(contacts));
0079: for  i = 1:length(contacts)
0080:     if isempty(contacts{i}) || startsWith(contacts{i}, 's') || ...
0081:             startsWith(contacts{i}, 'REF') || ...
0082:             startsWith(contacts{i}, 'Ref') || ...
0083:             startsWith(contacts{i}, 'EKG')
0084:         to_remove(i) = true;
0085:     end
0086: end
0087: 
0088: ttl_idx = startsWith(contacts, 'TTL'); %may need to change this to be more general
0089: 
0090: csc_names(to_remove) = [];
0091: csc_files(to_remove) = [];
0092: contacts(to_remove) = [];
0093: 
0094: % read in event information
0095: ev_fname = 'Events.nev';
0096: 
0097: [Timestamps, ~, TTLs, ~, ~, ~] = ...
0098:     Nlx2MatEV([csc_dir filesep ev_fname], [1 1 1 1 1], 1, 1, []);
0099: 
0100: Timestamps = Timestamps(TTLs>0);
0101: 
0102: if ~isempty(Timestamps)
0103:     has_ev = true;
0104:     timestamps = Timestamps(timestamps);
0105: else
0106:     try
0107:         ev_fname = 'ttl_Events.nev';
0108:         [Timestamps, ~, TTLs, ~, ~, ~] =  Nlx2MatEV([csc_dir filesep ev_fname], ...
0109:             [1 1 1 1 1], 1, 1, []);
0110:     
0111:         Timestamps = Timestamps(TTLs>0);
0112:         
0113:         if ~isempty(Timestamps)
0114:             has_ev = true;
0115:             timestamps = Timestamps(timestamps);
0116:         else
0117:             has_ev = false;
0118:         end
0119: 
0120:     catch
0121:         % if there are no timestamps in evfile, go ahead and generate off of TTL16
0122:         
0123:         [Timestamps, Samples, Header] = Nlx2MatCSC([csc_dir filesep csc_files(ttl_idx).name], ...
0124:         [1 0 0 0 1], ...
0125:         1, 1, 1);
0126: 
0127:         sr = str2double(Header{15}(20:end));
0128:         d = double(Timestamps(2:end)-Timestamps(1:end-1));
0129:         maxJump  = ceil(10^6./(sr-1))*512;
0130:         TimeStampPerSample =  nanmedian(d(d<maxJump))/512;
0131: 
0132:         % below assumes no large jumps in recording, adding in code here to
0133:         % check for this
0134: 
0135:         if any(diff(Timestamps)<0) % potential bug, or assume continuous
0136:             stamps = Timestamps(1):TimeStampPerSample:(Timestamps(end)+512*TimeStampPerSample-1);
0137:         else
0138:             stamps = arrayfun(@(i) Timestamps(i):TimeStampPerSample:(Timestamps(i)+512*TimeStampPerSample-1), 1:length(Timestamps), 'UniformOutput', false);
0139:             stamps = [stamps{:}];
0140:         end
0141: 
0142:         [~, locs] = findpeaks(abs(Samples(:)), 'MinPeakHeight', 3*10^4, 'MinPeakDistance', 1000);
0143:         ttl_Timestamps = stamps(locs);
0144: 
0145:         HeaderOut{1} = '######## Neuralynx';     %this is REQUIRED as header prefix
0146:         HeaderOut{2} = 'FileExport Mat2NlxEV unix-vers';    
0147:         HeaderOut{3} = ' matlab generated timestamps';    
0148:         
0149:         TTLs = ones(size(ttl_Timestamps));
0150: 
0151:         Mat2NlxEV([csc_dir filesep 'ttl_Events.nev'], ...
0152:                    0, ...
0153:                    1, ...
0154:                    1, ...
0155:                    length(ttl_Timestamps), ...
0156:                    [1 0 1 0 0 1], ...
0157:                    ttl_Timestamps, ...
0158:                    TTLs, ...
0159:                    HeaderOut' ); 
0160: 
0161:         timestamps = Timestamps(timestamps);
0162:     end
0163: end
0164: 
0165: ttl_idx = (startsWith(contacts,'TTL')); % get rid of all ttl at this point
0166: 
0167: csc_names(ttl_idx) = [];
0168: csc_files(ttl_idx) = [];
0169: contacts(ttl_idx) = [];
0170: 
0171: 
0172: epoched_dat = {};
0173: epoched_tim = {};
0174: 
0175: parfor f = 1:length(csc_files)
0176:     [Timestamps, Samples, Header] = Nlx2MatCSC([csc_dir filesep csc_files(f).name], ...
0177:         [1 0 0 0 1], ...
0178:         1, 1, 1);
0179:     
0180:     scale = str2double(Header{17}(14:end))*10^6; %multiply for muV
0181:     
0182:     % check for inverted signal
0183:     if strcmp(Header{22}(16:end), 'True') %inverted signal
0184:         scale = -scale; % invert
0185:     end
0186: 
0187:     sr = str2double(Header{15}(20:end));
0188:     d = double(Timestamps(2:end)-Timestamps(1:end-1));
0189:     maxJump  = ceil(10^6./(sr-1))*512;
0190:     TimeStampPerSample =  nanmedian(d(d<maxJump))/512;
0191:     
0192:     chan_dat = [];
0193:     
0194:     for e = 1:length(timestamps)
0195:         
0196:         % extra 2 seconds buffer to avoid missing data due to timestamp
0197:         % issues
0198:         timestamp_range_read = [timestamps(e)-TimeStampPerSample*sr*(pre_dur+2000)/1000 ...
0199:             timestamps(e)+TimeStampPerSample*sr*(post_dur+2000)/1000];
0200: 
0201:         timestamp_range = [timestamps(e)-TimeStampPerSample*sr*(pre_dur)/1000 ...
0202:             timestamps(e)+TimeStampPerSample*sr*(post_dur)/1000];
0203:         
0204:         [Timestamps_ev, Nsamp, Samples_ev, Header] = Nlx2MatCSC([csc_dir filesep csc_files(f).name], ...
0205:             [1 0 0 1 1], ...
0206:             1, 4, timestamp_range_read);
0207: 
0208:         to_keep = Nsamp == 512;
0209:         Timestamps_ev = Timestamps_ev(to_keep);
0210:         Samples_ev = Samples_ev(:, to_keep);
0211: 
0212:         mdiff = median(diff(Timestamps_ev));
0213: 
0214:         % check if there is an issue with timestamps
0215:         if ~all(diff(Timestamps_ev)==mdiff) % jitter bug, assume all TimeStampPerSample
0216: 
0217:             valid_idx = find(diff(Timestamps_ev)==mdiff,1,'first');
0218: 
0219:             for i = valid_idx:length(Timestamps_ev)-1
0220:                 Timestamps_ev(i+1) = Timestamps_ev(i)+mdiff;
0221:             end
0222: 
0223:             for i = 1:valid_idx % and reverse direction
0224:                 if valid_idx-i > 0
0225:                     Timestamps_ev(valid_idx-i) = Timestamps_ev(valid_idx-i+1)-mdiff;
0226:                 end
0227:             end
0228:         end
0229: 
0230:         time = nan(size(Samples_ev));
0231:         
0232:         for i = 1:size(Samples_ev,2)
0233:             ts = Timestamps_ev(i);
0234:             te = (512-1)*TimeStampPerSample + ts;
0235:             time(:,i) =  ts:TimeStampPerSample:te;
0236:         end
0237:         
0238:         % relative to this event, which timepoints to include
0239:         time = time(:) - timestamps(e);
0240:         include = time >= (timestamp_range(1) - timestamps(e)) & ...
0241:             time < (timestamp_range(2) - timestamps(e));
0242:         
0243:         if isempty(chan_dat) && any(include)
0244:             
0245:             chan_dat = nan(length(timestamps), ...
0246:                            sum(include), 'double'); % single for mem
0247:         end
0248:         
0249:         if any(include)
0250:            chan_dat(e,:) = scale*Samples_ev(include);
0251:         end
0252:         
0253:     end
0254:     
0255:     ds_dat = resample(chan_dat', 500, sr)';
0256: 
0257:     epoched_dat{f} = ds_dat;
0258:     epoched_tim{f} = 0:1/500:(1/500*(size(ds_dat,2)-1));
0259: 
0260: end
0261: 
0262: epoched_dat = cat(3,epoched_dat{:});
0263: epoched_dat = permute(epoched_dat, [3 1 2]);
0264: 
0265: time = epoched_tim{1}*1000;
0266: sr = 500;
0267: 
0268: end
```

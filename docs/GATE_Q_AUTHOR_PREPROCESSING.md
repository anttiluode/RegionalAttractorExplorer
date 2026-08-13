# Gate Q authors-preprocessing probe

## `TSS-main/tss/analysis/run_group_cl_phase.m`

```matlab
0001: function [polar_res, boot_res] = run_group_cl_phase() 
0002: %RUN_GROUP_CL_PHASE This function runs group-level analysis on phase data from different subjects
0003: % 
0004: % This function processes EEG data for a group of subjects to analyze phase consistency.
0005: % It reads and preprocesses the data, computes phase estimates, and generates various metrics.
0006: % The results are saved to CSV files for further analysis.
0007: % 
0008: % Outputs:
0009: %   polar_res - structure containing resultant vector lengths and angles for each subject
0010: %   boot_res - structure containing bootstrapped phase consistency metrics
0011: 
0012: 
0013: subjects = {'p16','p17','p18','p19','p20','UC004','UC005'};
0014: 
0015: read_ep = false;
0016: read_rest = false;
0017: read_cl = true;
0018: read_cl_control = false;
0019: 
0020: n = nan(1, length(subjects));
0021: n_control = nan(1, length(subjects));
0022: r = nan(1, length(subjects));
0023: r_control = nan(1, length(subjects));
0024: a = nan(1, length(subjects));
0025: a_control = nan(1, length(subjects));
0026: r_ci = nan(length(subjects), 2);
0027: r_boot = nan(10000, length(subjects));
0028: r_control_ci = nan(length(subjects), 2);
0029: r_control_boot = nan(10000, length(subjects));
0030: a_ci = nan(length(subjects), 2);
0031: a_control_ci = nan(length(subjects), 2);
0032: 
0033: phase_only = true; source_T = table; source_pred_T = table;
0034: 
0035: for s = 1:length(subjects)
0036:     
0037:     [csc_dir, img_dir] = get_dirs(subjects{s});
0038: 
0039:     is_control_subject = @(subj) strcmp(subj, 'p20') || startsWith(subj, 'UC');
0040: 
0041:     if is_control_subject(subjects{s})
0042:         has_cl = false;
0043:     else
0044:         has_cl = true;
0045:     end
0046: 
0047:     if has_cl
0048:         
0049:         dat = get_subj_eeg(subjects{s}, ...
0050:             csc_dir, ...
0051:             img_dir, ...
0052:             read_ep, ...
0053:             read_rest, ...
0054:             read_cl, ...
0055:             read_cl_control, ...
0056:             phase_only);
0057:     
0058:         dat = run_cl_preproc(dat.cl__epoched, ...
0059:             dat.sr, dat.contacts);
0060:     end
0061: 
0062:     % if subject has phase blind condition, show what the same phase
0063:     % distribution looks like for that subject
0064:  
0065:     if strcmp(subjects{s},'p19') || strcmp(subjects{s},'p17') || is_control_subject(subjects{s})
0066: 
0067:         read_cl_control = true;
0068:         read_cl = false;
0069:         dat_control = get_subj_eeg(subjects{s}, ...
0070:             csc_dir, ...
0071:             img_dir, ...
0072:             read_ep, ...
0073:             read_rest, ...
0074:             read_cl, ...
0075:             read_cl_control, ...
0076:             phase_only);
0077:         
0078:         dat_control = run_cl_preproc(dat_control.cl_control__epoched, ...
0079:             dat_control.sr, dat_control.contacts);
0080:         
0081:         read_cl = true;
0082:         read_cl_control = false;
0083:     end
0084: 
0085:     if has_cl
0086:         [est_phase{s}, pred_dat{s}] = pred_stim_phase(dat, 80);
0087:             % compute resultant angle length, can compute other metrics later
0088:         n(s) = size(est_phase{s},2);           % number of trials
0089:         rfn = @(x) abs(sum(exp(1i*x)))/n(s);
0090:         r(s) = rfn(est_phase{s}');
0091:         [r_ci(s,:), r_boot(:,s)] = bootci(10000, {rfn, est_phase{s}});
0092:         a(s) = angle(sum(exp(1i*est_phase{s}),2));
0093:         afn = @(x) angle(sum(exp(1i*x)));
0094:         a_ci(s,:) = bootci(10000, {afn, est_phase{s}});
0095: 
0096:     end
0097:     
0098:     % if this subj has control stim
0099:     if strcmp(subjects{s},'p19') || strcmp(subjects{s},'p17') || strcmp(subjects{s},'p20') || ...
0100:         startsWith(subjects{s},'UC')
0101:         [control_est_phase{s}, control_pred_dat{s}] = pred_stim_phase(dat_control, 80);
0102:         
0103:         n_control(s) = size(control_est_phase{s},2);           % number of trials
0104:         rfn = @(x) abs(sum(exp(1i*x)))/n_control(s);
0105:         r_control(s) = rfn(control_est_phase{s}');
0106:         [r_control_ci(s,:), r_control_boot(:,s)] = bootci(10000, {rfn, control_est_phase{s}});
0107:         a_control(s) = angle(sum(exp(1i*control_est_phase{s}),2));
0108:         a_control_ci(s,:) = bootci(10000, {afn, control_est_phase{s}});
0109:     end
0110: 
0111:     if is_control_subject(subjects{s})
0112:         has_cl = false;
0113:         has_control = true;
0114:     else
0115:         has_cl = true;
0116:         if strcmp(subjects{s},'p17') || strcmp(subjects{s},'p19')
0117:             has_control = true;
0118:         else
0119:             has_control = false;
0120:         end
0121:     end
0122: 
0123:     % Write out source data files for both actual and predicted data
0124:     % This includes data for both the CL (closed-loop) and PB (phase-blind) conditions
0125: 
0126:     % actual dat
0127:     T = []; control_T = [];
0128:     if has_cl
0129:         tmp = cat(1,dat.trial{:});
0130: 
0131:         varNames = arrayfun(@(x) sprintf('Timepoint_%d', x), 1:size(tmp, 2), 'UniformOutput', false);
0132:         T = array2table(tmp, 'VariableNames', varNames);
0133: 
0134:         trialNumbers = (1:size(tmp, 1))';
0135:         T.SubjectID = repmat({subjects{s}}, size(tmp, 1), 1);
0136:         T.Condition = repmat({'CL'}, size(tmp, 1), 1);
0137:         T.TrialNumber = trialNumbers;
0138:         T = [T(:, end-2:end), T(:, 1:end-3)];
0139: 
0140:         source_T = vertcat(source_T, T);
0141:     end
0142: 
0143:     if has_control
0144:         
0145:         tmp = cat(1,dat_control.trial{:});
0146: 
0147:         varNames = arrayfun(@(x) sprintf('Timepoint_%d', x), 1:size(tmp, 2), 'UniformOutput', false);
0148:         control_T = array2table(tmp, 'VariableNames', varNames);
0149: 
0150:         trialNumbers = (1:size(tmp, 1))';
0151:         control_T.SubjectID = repmat({subjects{s}}, size(tmp, 1), 1);
0152:         control_T.Condition = repmat({'PB'}, size(tmp, 1), 1);
0153:         control_T.TrialNumber = trialNumbers;
0154:         control_T = [control_T(:, end-2:end), control_T(:, 1:end-3)];
0155: 
0156:         source_T = vertcat(source_T, control_T);
0157:     end
0158: 
0159:     % predicted dat for CL
0160:     T = []; control_T = [];
0161:     if has_cl
0162:         tmp = cat(1,pred_dat{s}.trial{:});
0163: 
0164:         varNames = arrayfun(@(x) sprintf('Timepoint_%d', x), 1:size(tmp, 2), 'UniformOutput', false);
0165:         T = array2table(tmp, 'VariableNames', varNames);
0166: 
0167:         trialNumbers = (1:size(tmp, 1))';
0168:         T.SubjectID = repmat({subjects{s}}, size(tmp, 1), 1);
0169:         T.Condition = repmat({'CL'}, size(tmp, 1), 1);
0170:         T.TrialNumber = trialNumbers;
0171:         T = [T(:, end-2:end), T(:, 1:end-3)];
0172: 
0173:         source_pred_T = vertcat(source_pred_T, T);
0174:     end
0175: 
0176:     if has_control
0177:         
0178:         tmp = cat(1,control_pred_dat{s}.trial{:});
0179: 
0180:         varNames = arrayfun(@(x) sprintf('Timepoint_%d', x), 1:size(tmp, 2), 'UniformOutput', false);
0181:         control_T = array2table(tmp, 'VariableNames', varNames);
0182: 
0183:         trialNumbers = (1:size(tmp, 1))';
0184:         control_T.SubjectID = repmat({subjects{s}}, size(tmp, 1), 1);
0185:         control_T.Condition = repmat({'PB'}, size(tmp, 1), 1);
0186:         control_T.TrialNumber = trialNumbers;
0187:         control_T = [control_T(:, end-2:end), control_T(:, 1:end-3)];
0188: 
0189:         source_pred_T = vertcat(source_pred_T, control_T);
0190:     end
0191: 
0192: end
0193:    
0194: writetable(source_T, 'source_table_cldat.csv');
0195: writetable(source_pred_T, 'source_pred_table_cldat.csv');
0196: 
0197: % organize and write phase information
0198: 
0199: source_phase_T = []; 
0200: 
0201: for s = 1:length(subjects)
0202:     if s < 5
0203:         if ~isempty(est_phase{s}) % has cl
0204: 
0205:             tmp = est_phase{s}';
0206:             T = array2table(tmp, 'VariableNames', {'Theta Phase'});
0207:             trialNumbers = (1:size(tmp, 1))';
0208:             T.SubjectID = repmat({subjects{s}}, size(tmp, 1), 1);
0209:             T.Condition = repmat({'CL'}, size(tmp, 1), 1);
0210:             T.TrialNumber = trialNumbers;
0211:             T = [T(:, end-2:end), T(:, 1:end-3)];
0212: 
0213:             source_phase_T = vertcat(source_phase_T, T);
0214:         end
0215:     end
0216:     if ~isempty(control_est_phase{s})
0217: 
0218:         tmp = control_est_phase{s}';
0219:         T = array2table(tmp, 'VariableNames', {'Theta Phase'});
0220:         trialNumbers = (1:size(tmp, 1))';
```

## `TSS-main/tss/analysis/run_group_ged.m`

```matlab
0001: function [res, control_res, dat, control_dat, power_table, p2p_table] = run_group_ged()
0002: % RUN_GROUP_GED runs the group generalized eigenvalue decomposition analysis
0003: %
0004: % Outputs:
0005: %   res: results from the group GED analysis
0006: %   control_res: results from the control group GED analysis
0007: %   dat: data from the group
0008: %   control_dat: data from the control group
0009: %   power_table: table of power data
0010: %   p2p_table: table of peak-to-peak amplitude data
0011: 
0012: subjects = {'p16','p17','p18','p19','p20','UC004','UC005'};
0013: 
0014: for s = 1:length(subjects)
0015: 
0016:     if ~(strcmp(subjects{s},'p20') || strcmp(subjects{s},'UC004') || ...
0017:             strcmp(subjects{s},'UC005'))
0018: 
0019:         is_control = false;
0020: 
0021:         [stats{s}, dat{s}, stats_early{s}, stats_late{s}, p2p_early{s}, p2p_late{s}] = run_subj_analysis(subjects{s}, is_control);
0022: 
0023:     end
0024: 
0025:     if strcmp(subjects{s},'p17') || strcmp(subjects{s},'p19') || ...
0026:             strcmp(subjects{s},'p20') || strcmp(subjects{s},'UC004') || ...
0027:             strcmp(subjects{s},'UC005')
0028: 
0029:         is_control = true;
0030: 
0031:         [control_stats{s}, control_dat{s}, control_stats_early{s}, control_stats_late{s}, control_p2p_early{s}, control_p2p_late{s}] = run_subj_analysis(subjects{s}, is_control);
0032: 
0033:     end
0034: 
0035: end
0036: 
0037: res.stats = stats;
0038: res.stats_early = stats_early;
0039: res.stats_late = stats_late;
0040: res.p2p_early = p2p_early;
0041: res.p2p_late = p2p_late;
0042: 
0043: control_res.stats = control_stats;
0044: control_res.stats_early = control_stats_early;
0045: control_res.stats_late = control_stats_late;
0046: control_res.p2p_early = control_p2p_early;
0047: control_res.p2p_late = control_p2p_late;
0048: 
0049: % convert to long format for a simple LME model
0050: 
0051: Y = []; Y_prev = []; F = []; C = [];S = [];N=[];
0052: 
0053: for s = 1:length(subjects)
0054:     for f = 1:45
0055: 
0056:         if s <= length(dat)
0057:             if ~isempty(dat{s})
0058:                 Y_tmp = squeeze(dat{s}.powspctrm(:,:,f));
0059:                 Y = cat(1,Y,Y_tmp);
0060:                 Y_prev = cat(1, Y_prev, [NaN; Y_tmp(1:end-1)]);
0061:                 S = cat(1,S,s*ones(size(squeeze(dat{s}.powspctrm(:,:,f)))));
0062:                 C = cat(1,C,1*ones(size(squeeze(dat{s}.powspctrm(:,:,f)))));
0063:                 F = cat(1,F,f*ones(size(squeeze(dat{s}.powspctrm(:,:,f)))));
0064:                 tmp = (1:size(dat{s}.powspctrm,1))';
0065:                 tmp = (tmp-mean(tmp))/max(tmp); % middle of the period is t0
0066:                 N = cat(1,N,tmp);
0067:             end
0068:         end
0069:         if ~isempty(control_dat{s}) && s <= length(control_dat)
0070: 
0071:             Y_tmp = squeeze(control_dat{s}.powspctrm(:,:,f));
0072:             Y = cat(1,Y,Y_tmp);
0073:             Y_prev = cat(1, Y_prev, [NaN; Y_tmp(1:end-1)]);
0074: 
0075:             S = cat(1,S,s*ones(size(squeeze(control_dat{s}.powspctrm(:,:,f)))));
0076:             C = cat(1,C,2*ones(size(squeeze(control_dat{s}.powspctrm(:,:,f)))));
0077:             F = cat(1,F,f*ones(size(squeeze(control_dat{s}.powspctrm(:,:,f)))));
0078:             tmp = (1:size(control_dat{s}.powspctrm,1))';
0079:             tmp=(tmp-mean(tmp))/max(tmp);
0080:             N = cat(1,N,tmp);
0081:         end
0082:     end
0083: end
0084: 
0085: power_table = table(Y, Y_prev, categorical(S),categorical(C),categorical(F),N, ...
0086:     'VariableNames', {'Power','Power_lag1','Subject','Condition','Frequency','Time'});
0087: 
0088: power_table.Condition = categorical(power_table.Condition);
0089: power_table.Frequency = categorical(power_table.Frequency);
0090: power_table.Subject = categorical(power_table.Subject);
0091: 
0092: 
0093: % similar table, but for amplitude for p2p
0094: Y = []; EL = []; C = [];S = [];N=[];
0095: 
0096: for s = 1:length(subjects)
0097:         if s <= length(dat)
0098:             if ~isempty(dat{s})
0099:                 Y_tmp = squeeze(res.p2p_early{s})';
0100:                 Y = cat(1,Y,Y_tmp);
0101:                 S = cat(1,S,s*ones(size(res.p2p_early{s}')));
0102:                 C = cat(1,C,1*ones(size(res.p2p_early{s}')));
0103:                 EL = cat(1,EL,ones(size(res.p2p_early{s}')));
0104: 
0105:                 tmp = (1:size(res.p2p_early{s},2))';
0106:                 tmp = (tmp-mean(tmp))/max(tmp); % middle of the period is t0
0107:                 N = cat(1,N,tmp);
0108: 
0109:                 Y_tmp = squeeze(res.p2p_late{s})';
0110:                 Y = cat(1,Y,Y_tmp);
0111:                 S = cat(1,S,s*ones(size(res.p2p_late{s}')));
0112:                 C = cat(1,C,1*ones(size(res.p2p_late{s}')));
0113:                 EL = cat(1,EL,0*ones(size(res.p2p_late{s}')));
0114: 
0115:                 tmp = (1:size(res.p2p_late{s},2))';
0116:                 tmp = (tmp-mean(tmp))/max(tmp); % middle of the period is t0
0117:                 N = cat(1,N,tmp);
0118:             end
0119:         end
0120:         if ~isempty(control_dat{s}) && s <= length(control_dat)
0121: 
0122:                 Y_tmp = squeeze(control_res.p2p_early{s})';
0123:                 Y = cat(1,Y,Y_tmp);
0124:                 S = cat(1,S,s*ones(size(control_res.p2p_early{s}')));
0125:                 C = cat(1,C,2*ones(size(control_res.p2p_early{s}')));
0126:                 EL = cat(1,EL,ones(size(control_res.p2p_early{s}')));
0127: 
0128:                 tmp = (1:size(control_res.p2p_early{s},2))';
0129:                 tmp = (tmp-mean(tmp))/max(tmp); % middle of the period is t0
0130:                 N = cat(1,N,tmp);
0131: 
0132:                 Y_tmp = squeeze(control_res.p2p_late{s})';
0133:                 Y = cat(1,Y,Y_tmp);
0134:                 S = cat(1,S,s*ones(size(control_res.p2p_late{s}')));
0135:                 C = cat(1,C,2*ones(size(control_res.p2p_late{s}')));
0136:                 EL = cat(1,EL,0*ones(size(control_res.p2p_late{s}')));
0137: 
0138:                 tmp = (1:size(control_res.p2p_late{s},2))';
0139:                 tmp = (tmp-mean(tmp))/max(tmp); % middle of the period is t0
0140:                 N = cat(1,N,tmp);
0141: 
0142:         end
0143: end
0144: 
0145: p2p_table = table(Y, categorical(EL), categorical(S),categorical(C),N, ...
0146:     'VariableNames', {'Amplitude','WaveformEL','Subject','Condition','Time'});
0147: 
0148: lme = fitlme(p2p_table(p2p_table.WaveformEL==categorical(1),:),'Amplitude~Condition*Time+(Condition*Time|Subject)', ...
0149:         'DummyVarCoding','effects', ...
0150:         'CovariancePattern','full');
0151: 
0152: lme_reduced = fitlme(p2p_table(p2p_table.WaveformEL==categorical(1),:),'Amplitude~Condition+Time+(Condition*Time|Subject)', ...
0153:     'DummyVarCoding','effects', ...
0154:     'CovariancePattern','full');
0155: 
0156:     anova_res = anova(lme,'DFMethod','Satterthwaite');
0157:     anova_res.FStat
0158:     anova_res.pValue
0159:     compare(lme_reduced, lme)
0160: 
0161: lme = fitlme(p2p_table(p2p_table.WaveformEL==categorical(0),:),'Amplitude~Condition*Time+(Condition*Time|Subject)', ...
0162:         'DummyVarCoding','effects', ...
0163:         'CovariancePattern','full');
0164: 
0165: lme_reduced = fitlme(p2p_table(p2p_table.WaveformEL==categorical(0),:),'Amplitude~Condition+Time+(Condition*Time|Subject)', ...
0166:     'DummyVarCoding','effects', ...
0167:     'CovariancePattern','full');
0168: 
0169: anova_res = anova(lme,'DFMethod','Satterthwaite');
0170: anova_res.FStat
0171: anova_res.pValue
0172: compare(lme_reduced, lme)
0173: 
0174: % increases during closed-loop
0175: for f = 1:45
0176: 
0177:     % don't need dummyvarcoding for continuous X only
0178:     lme = fitlme(power_table(power_table.Frequency==categorical(f) & power_table.Condition == categorical(1),:),'Power~Time+(Time|Subject)');
0179: 
0180:     cl_mdl{f} = lme;
0181:     cl_anova_res{f} = anova(lme,'DFMethod','Satterthwaite');
0182:     cl_Fs(f,:) = cl_anova_res{f}.FStat;
0183:     cl_pValues(f,:) = cl_anova_res{f}.pValue;
0184:     fEffects(f,:) = lme.fixedEffects;
0185:     fCI(f,:,:) = lme.coefCI;
0186: 
0187: end
0188: 
0189: anova_res= {};
0190: % interaction
0191: for f = 1:45
0192: 
0193:     lme = fitlme(power_table(power_table.Frequency==categorical(f),:),'Power~Condition*Time+(Condition*Time|Subject)', ...
0194:         'DummyVarCoding','effects', ...
0195:         'CovariancePattern','full');
0196: 
0197:     lme_reduced = fitlme(power_table(power_table.Frequency==categorical(f),:),'Power~Condition+Time+(Condition*Time|Subject)', ...
0198:         'DummyVarCoding','effects', ...
0199:         'CovariancePattern','full');
0200: 
0201:     anova_res{f} = anova(lme,'DFMethod','Satterthwaite');
0202:     intxn_Fs(f,:) = anova_res{f}.FStat;
0203:     intxn_pValues(f,:) = anova_res{f}.pValue;
0204: 
0205:     res = compare(lme_reduced, lme);
0206: 
0207:     pval(f) = res.pValue(end);
0208:     stat(f) = res.LRStat(end);
0209: 
0210:     B(f,:,1) = lme.fixedEffects;
0211:     CI(f,:,:) = lme.coefCI;
0212: end
0213: 
0214: plot_theta_fig2(stats, ...
0215:                 control_stats, ...
0216:                 power_table, ...
0217:                 fEffects, ...
0218:                 fCI, ...
0219:                 cl_pValues, ...
0220:                 B, ...
...
0224: end
0225: 
0226: function [stats, dat, stats_early, stats_late, p2p_early, p2p_late] = run_subj_analysis(subject, is_control, is_clean)
0227: 
0228: 
0229: if ~is_control
0230:     read_ep = false;
0231:     read_rest = false;
0232:     read_cl = true;
0233:     read_cl_control = false;
0234: else
0235:     read_ep = false;
0236:     read_rest = false;
0237:     read_cl = false;
0238:     read_cl_control = true;
0239: end
0240: 
0241: [clean_dat, ged_res, dat_lf] = get_cleaned_dat(subject, ...
0242:     read_ep, ...
0243:     read_rest, ...
0244:     read_cl, ...
0245:     read_cl_control, ...
0246:     is_control);
0247: 
0248: stim = get_stim_info(subject);
0249: if ~strcmp(subject,'p20')
0250:     is_stim = strcmp(clean_dat.label, stim.anode) | ...
0251:         strcmp(clean_dat.label, stim.cathode);
0252: else
0253:     is_stim = strcmp(clean_dat.label, 'D''10') | ...
0254:         strcmp(clean_dat.label, 'D''11');
0255: end
0256: 
0257: phase_chan = get_phase_chan(subject, clean_dat.label);
0258: clean_freq = get_clean_fourier(clean_dat);
0259: 
0260: k_thr = 5; %5 kurtosis for trial exclusion, no trial exclusion
0261: z_thr = 5; %5 z score for trial exclusion,  no trial exclusion
0262: a_thr = 500;
0263: lp_flag = false;
0264: 
0265: [res, trial_idx] = run_cl_ep_preproc(dat_lf, ...
0266:                         phase_chan, ...
0267:                         k_thr, ...
0268:                         z_thr, ...
0269:                         a_thr, ...
0270:                         lp_flag);
0271: 
0272: [p2p_early, max_t_early, min_t_early] = compute_p2p(res, [.015 .05]);
...
0316: cfg.avgovertime = 'yes'; %'yes';
0317: cfg.latency = [0.1 0.5];
0318: dat = ft_selectdata(cfg, clean_freq);
0319: 
0320: end
0321: 
0322: function [clean_dat, ged_res, dat_lf] = get_cleaned_dat(subject, read_ep, read_rest, read_cl, ...
0323:     read_cl_control, is_control)
0324: 
0325: if ~exist('is_control','var')
0326:     is_control = false;
0327: end
0328: 
0329: if is_control
0330:     fname = 'ged_cl_control_dat.mat';
...
0333: end
0334: 
0335: proc_dir = '/path/to/proc_data/';
0336: 
0337: if exist([proc_dir subject '/' fname],'file')
0338:     if ~is_control
0339:         load([proc_dir subject '/' fname], ...
0340:             'clean_dat','ged_res','dat_lf');
0341:     else
0342:         load([proc_dir subject '/' fname], ...
0343:             'clean_control_dat','ged_control_res','dat_control_lf');
0344:     end
0345: else
0346: 
0347:     [csc_dir, img_dir] = get_dirs(subject);
0348: 
0349:     if is_control
0350:         read_cl_control = true;
0351:         read_cl = false;
0352:     end
0353: 
0354:     if ~is_control
0355:         dat = get_subj_eeg(subject, ...
0356:             csc_dir, ...
0357:             img_dir, ...
0358:             read_ep, ...
0359:             read_rest, ...
0360:             read_cl, ...
0361:             read_cl_control, ...
0362:             'all');
0363:     else
0364:         dat_control = get_subj_eeg(subject, ...
0365:             csc_dir, ...
0366:             img_dir, ...
0367:             read_ep, ...
0368:             read_rest, ...
```

## `TSS-main/tss/preprocessing/run_cl_ep_preproc.m`

```matlab
0001: function [timelock, trial_exclude] = run_cl_ep_preproc(cl_epoched_dat, ...
0002:     chansel, kthr, zthr, athr, lp_flag)
0003: %RUN_CL_EP_PREPROC runs preprocessing for closed-loop stimulation evoked
0004: %  potential data
0005: %
0006: % Input:
0007: %   cl_epoched_dat: closed-loop epoched data
0008: %   chansel: channel selection
0009: %   kthr: kurtosis threshold
0010: %   zthr: z-score threshold
0011: %   athr: amplitude threshold
0012: %   lp_flag: whether to low-pass filter
0013: %
0014: % Output:
0015: %   timelock: timelock data
0016: %   trial_exclude: trial indices to exclude
0017: %
0018: 
0019: dat = run_preproc(cl_epoched_dat, chansel, lp_flag);
0020: 
0021: time_shift = 1;
0022: % adjust timing
0023: for t = 1:length(dat.time)
0024:     dat.time{t} = dat.time{t} - time_shift;
0025: end
0026: 
0027: % and timelock
0028: cfg = [];
0029: cfg.latency = [-.4 0.8];
0030: cfg.keeptrials = 'yes';
0031: timelock = ft_timelockanalysis(cfg, dat);
0032: 
0033: a = max(abs(timelock.trial(:,:, timelock.time > .05 | timelock.time < -.05)), [], 3);
0034: k = kurtosis(timelock.trial(:,:,timelock.time > .05), [], 3);
0035: z = zscore(squeeze(timelock.trial));
0036: 
0037: cfg = [];
0038: trial_exclude = all(a < athr & k < kthr & ~any(abs(z(:,timelock.time > -Inf)) > zthr,2),2);
0039: cfg.trials = trial_exclude;
0040: timelock = ft_selectdata(cfg, timelock);
0041: 
0042: cfg = [];
0043: cfg.baseline = [-.05 -0.01];
0044: cfg.parameter = 'trial';
0045: cfg.keeptrials = 'yes';
0046: timelock = ft_timelockbaseline(cfg, timelock);
0047: 
0048: end
0049: 
0050: 
0051: function datout = run_preproc(datin, chansel, lp_flag)
0052: % RUN_PREPROC runs preprocessing for the input data
0053: %
0054: % Input:
0055: %   datin: continuous data structure
0056: %   chansel: channel selection
0057: %   lp_flag: whether to low-pass filter
0058: %
0059: % Output:
0060: %   datout: continuous data structure with preprocessing applied
0061: %
0062: 
0063: cfg = [];
0064: cfg.channel = chansel;
0065: datin = ft_selectdata(cfg, datin);
0066: 
0067: cfg = [];
0068: cfg.latency = [datin.time{1}(1) datin.time{1}(end-1)]; %samples for dft -1
0069: datout = ft_selectdata(cfg, datin);
0070: 
0071: cfg = [];
0072: cfg.dftfilter = 'yes';
0073: cfg.dftfreq = [60 120 180];
0074: cfg.dftbandwidth = [.5 .5 .5];
0075: cfg.dftneighbourwidth = [1.5 1.5 1.5];
0076: 
0077: datout = ft_preprocessing(cfg, datout);
0078: 
0079: if lp_flag
0080: 
0081:     cfg = [];
0082:     cfg.lpfilter = 'yes';
0083:     cfg.lpfreq = 80;
0084:     cfg.lpfiltord = 10;
0085: 
0086:     datout = ft_preprocessing(cfg, datout);
0087: 
0088: end
0089: 
0090: end
```

## `TSS-main/tss/utils/get_stim_info.m`

```matlab
0001: function stim = get_stim_info(subjid)
0002: %GET_STIM_INFO returns stimulation info for a given subject
0003: %
0004: % Input:
0005: %   subjid: subject ID
0006: %
0007: % Output:
0008: %   stim: structure containing stimulation info
0009: %
0010: 
0011: switch subjid
0012:     
0013:     case 'p09'
0014:         stim.anode = 'B10'; % -
0015:         stim.cathode = 'B11'; %+ 
0016:         stim.phase = 'B1-B2';
0017:     case 'p11'
0018:         stim.anode = 'D10'; % -
0019:         stim.cathode = 'D11'; % +
0020:         stim.phase = 'D1-D2';
0021:     case 'p15'      
0022:         stim.anode = 'P7'; % -
0023:         stim.cathode = 'P8'; % +
0024:         stim.phase = 'P2';
0025:     case 'p16'
0026:         stim.anode = 'RB8'; % -
0027:         stim.cathode = 'RB9'; % +
0028:         stim.phase = 'RB1';
0029:     case 'p17'
0030:         stim.anode = 'D7'; % -
0031:         stim.cathode = 'D8'; % +
0032:         stim.phase = 'D1';
0033:     case 'p18'
0034:         stim.anode = 'AH6'; % -
0035:         stim.cathode = 'AH7'; % +
0036:         stim.phase = 'AH2';
0037:     case 'p19'
0038:         stim.anode = 'D9';
0039:         stim.cathode = 'D10';
0040:         stim.phase = 'D1';
0041:     case 'p20'
0042:         stim.anode = {'D9','C1','C''1','D''10'}; % -
0043:         stim.cathode = {'D10','C2','C''2','D''11'}; % +
0044:         stim.phase = 'C''1'; % control only
0045:     case 'UC004'
0046:         stim.anode = 'RHT6';
0047:         stim.cathode = 'RHT7';
0048:         stim.phase = 'RHH1'; %control only
0049:     case 'UC005'
0050:         stim.anode = 'RHB7';
0051:         stim.cathode = 'RHB8';
0052:         stim.phase = 'RHB1'; %control only
0053: end
0054: 
0055: end
0056: 
```

## `TSS-main/tss/analysis/run_bosc_fooof.m`

```matlab
0060:         ps = log10(ps');
0061: 
0062:         [ap_params, ap_ps] = robust_ap_fit(freqs, ps, [nan, ps(1), 0]);
0063: 
0064:         ap_ps = 10.^ap_ps;
0065: 
0066:         [powthresh,durthresh] = BOSC_thresholds(fs, ...
0067:             0.95, ... %
0068:             3, ...
0069:             F, ...
0070:             ap_ps);
0071: 
0072:         is_osc = nan(size(B));
0073:         for fr = 1:size(B,1)
```

## `TSS-main/tss/analysis/run_ged_phaselocked.m`

```matlab
0007: %       dat - FieldTrip data structure containing the trial data
0008: %
0009: %   Output:
0010: %       clean_dat - FieldTrip data structure with phase-locked components removed
0011: %       ged_res - Structure containing GED results including eigenvectors, eigenvalues, and component maps
0012: 
0013: % Define threshold for eigenvalues
0014: eval_threshold = 0.01;
0015: 
0016: % time_idx - logical index array indicating the timepoints to build covariance matrices from.
0017: % This should include code to read time from dat and reflect timepoints relative to each event.
0018: % Here, it selects timepoints between -0.050 and 0.800 seconds relative to each event.
0019: 
0020: time_idx = ((dat.time{1}-1) > - .050) & ((dat.time{1}-1) < 0.800);
0021: 
0022: raw = cat(3, dat.trial{:});
0023: % could add data cleaning on single trial measures here
0024: 
...
0087: ged_res.compsign = compsign;
0088: ged_res.mapsign = mapsign;
0089: 
0090: % create fieldtrip structure with components - and remove phase-locked
0091: % components with evals >  .01
0092: 
0093: to_remove = diag(evals) > eval_threshold;
0094: 
0095: cleaned  = zeros(size(raw));
0096: for t = 1:size(raw,3)
0097:     cleaned(:,:,t) = pinv(evecs(:,~to_remove)')*evecs(:,~to_remove)'*raw(:,:,t);
0098: end
0099: 
0100: clean_dat = dat;
```

## `TSS-main/tss/analysis/run_group_bosc.m`

```matlab
0006: %       res = run_group_bosc()
0007: %   Output:
0008: %       res - A structure array containing the processed EEG data for each subject.
0009: 
0010: subjects = {'UC004','UC005','p16','p17','p18','p19','p20'};
0011: 
0012: read_ep = false;
0013: read_rest = true;
0014: read_cl = false;
0015: read_cl_control = false;
0016: 
0017: chantype = 'phase';
0018: 
0019: % Preallocate arrays for performance
0020: res_pre(length(subjects)) = struct();
0021: res_control_pre(length(subjects)) = struct();
0022: 
...
0025:     
0026:     [csc_dir, img_dir] = get_dirs(subjects{subjIdx});
0027: 
0028:     dat = get_subj_eeg(subjects{subjIdx}, ...
0029:         csc_dir, ...
0030:         img_dir, ...
0031:         read_ep, ...
0032:         read_rest, ...
0033:         read_cl, ... 
0034:         read_cl_control);
0035:     
0036:     phase_chan = get_phase_chan(subjects{subjIdx}, dat.contacts);
0037:     stim_info = get_stim_info(subjects{subjIdx});
0038: 
0039:     if isequal(chantype,'stim')
0040:         if ~strcmp(subjects{subjIdx},'p20')
0041:             phase_chan = {stim_info.anode stim_info.cathode};
0042:         else
0043:             phase_chan = {'D''10' 'D''11'};
0044:         end
```

## `TSS-main/tss/analysis/run_group_ep.m`

```matlab
0001: function res = run_group_ep()
0002: %RUN_GROUP_EP Processes EEG data for a group of subjects and performs statistical analysis.
0003: %
0004: %   This function reads EEG data for a group of subjects, preprocesses the data,
0005: %   and performs statistical analysis on the preprocessed data. The function
0006: %   supports different conditions and subjects, including control subjects.
0007: %
0008: %   Outputs:
0009: %       res - A structure containing the following fields:
0010: %           timelock_pre - Preprocessed data for the pre-stimulation period.
0011: %           timelock_post - Preprocessed data for the post-stimulation period.
...
0019: %
0020: %   Example:
0021: %       res = run_group_ep();
0022: 
0023: subjects = {'p16','p17','p18','p19','p20','UC004','UC005'};
0024: 
0025: read_ep = true;
0026: read_rest = false;
0027: read_cl = false;
0028: read_cl_control = false;
0029: 
0030: timelock_pre = cell(1,length(subjects));
0031: timelock_post = cell(1,length(subjects));
0032: con_timelock_pre = cell(1,length(subjects));
0033: con_timelock_post = cell(1,length(subjects));
0034: stats = cell(1,length(subjects));
0035: con_stats = cell(1,length(subjects));
...
0038: 
0039:     [csc_dir, img_dir] = get_dirs(subjects{s});
0040: 
0041:     dat = get_subj_eeg(subjects{s}, ...
0042:         csc_dir, ...
0043:         img_dir, ...
0044:         read_ep, ...
0045:         read_rest, ...
0046:         read_cl, ...
0047:         read_cl_control);
0048: 
0049:     if strcmp(subjects{s}, 'p16') % Subject p16 has inverted pre-stimulation epochs due to incorrect recording settings.
0050:         dat.ep_pre_epoched = -dat.ep_pre_epoched;
0051:     end
0052: 
0053: 
0054:     if strcmp(subjects{s}, 'p19') % Subject p19 has inverted pre- and post-stimulation epochs due to incorrect recording settings.
0055:         dat.ep_pre_epoched = -dat.ep_pre_epoched;
0056:         dat.ep_post_epoched = -dat.ep_post_epoched;
0057:     end
0058: 
0059:     stim_info = get_stim_info(subjects{s});
0060:     phase_chan = {stim_info.phase};
0061: 
0062:     k_thr = 5; %5 kurtosis for trial exclusion, no trial exclusion
0063:     z_thr = 5; %5 z score for trial exclusion,  no trial exclusion
0064:     a_thr = 500;
0065: 
0066:     if strcmp(subjects{s},'p17')
```

## `TSS-main/tss/analysis/run_group_rest_coherence.m`

```matlab
0003: % This function computes coherence and connectivity measures for a group of subjects
0004: % during resting state, both pre- and post-stimulation, and performs statistical analysis.
0005: %   Detailed explanation goes here
0006: 
0007: subjects = {'p16','p17','p18','p19','p20','UC004','UC005'};
0008: 
0009: read_ep = false;
0010: read_rest = true;
0011: read_cl = false;
0012: read_cl_control = false;
0013: 
0014: for s = 1:length(subjects)
0015:     
0016:     [csc_dir, img_dir] = get_dirs(subjects{s});
0017:     
0018:     dat = get_subj_eeg(subjects{s}, ...
0019:         csc_dir, ...
0020:         img_dir, ...
0021:         read_ep, ...
0022:         read_rest, ...
0023:         read_cl, ...
0024:         read_cl_control, ...
0025:         'all'); 
0026:     
0027:     
0028:     if ~isequal(subjects{s},'p20') && ~startsWith(subjects{s},'UC')% control only
0029:        
0030:         stim_info = get_stim_info(subjects{s});
0031:         if ~strcmp(subjects{s},'p20')
0032:             is_stim = strcmp(dat.contacts, stim_info.anode) | ...
0033:                 strcmp(dat.contacts, stim_info.cathode);
0034:         else
0035:             is_stim = strcmp(dat.contacts, 'D''10') | ...
0036:                 strcmp(dat.contacts, 'D''11');
0037:         end
0038: 
0039:         phase_chan = get_phase_chan(subjects{s}, dat.contacts);
0040:         is_phase = strcmp(dat.contacts, phase_chan);
0041: 
0042:         [freq_pre, freq_post] = compute_fourier(dat, ...
0043:                                                 'rest_pre_epoched', ...
0044:                                                 'rest_post_epoched', ...
0045:                                                 is_stim, ...
0046:                                                 is_phase);
0047: 
0048:         
0049:         % compute phase locking and stats -- stim channel to entire brain
0050:         [pre_conn{s}, post_conn{s}] = get_conn(freq_pre, freq_post, is_stim);
0051:         is_phase = endsWith(pre_conn{s}.label, phase_chan{:}) | startsWith(pre_conn{s}.label, [phase_chan{:} '-']);
0052:         
0053:         phase_chan = get_phase_chan(subjects{s}, freq_pre.label);
0054:         is_phase = endsWith(pre_conn{s}.label, phase_chan{:}) | startsWith(pre_conn{s}.label, [phase_chan{:} '-']);
0055: 
0056:         [stats{s}, theta_stats{s}] = run_all_stats(pre_conn{s}, post_conn{s}, pre_conn{s}.label(is_phase));
0057:         
0058:         measure = 'pli';
0059:         stim_chans = freq_pre.label(is_stim);
0060:         
0061:     end
0062:     
0063:     if strcmp(subjects{s}, 'p17') || strcmp(subjects{s}, 'p19') || strcmp(subjects{s}, 'p20') || ...
0064:             strcmp(subjects{s}, 'UC004') || strcmp(subjects{s}, 'UC005')% has control
0065:         
0066:         stim_info = get_stim_info(subjects{s});
0067:         if ~strcmp(subjects{s},'p20')
0068:             is_stim = strcmp(dat.contacts, stim_info.anode) | ...
0069:                 strcmp(dat.contacts, stim_info.cathode);
0070:         else
0071:             is_stim = strcmp(dat.contacts, 'D''10') | ...
0072:                 strcmp(dat.contacts, 'D''11');
0073:         end
0074: 
0075:         phase_chan = get_phase_chan(subjects{s}, dat.contacts);
0076:         is_phase = strcmp(dat.contacts, phase_chan);
0077: 
0078:         [freq_pre, freq_post] = compute_fourier(dat, ...
0079:                                                 'rest_control_pre_epoched', ...
0080:                                                 'rest_control_post_epoched', ...
0081:                                                 is_phase, ...
0082:                                                 is_stim);
0083:         
0084:         [con_pre_conn{s}, con_post_conn{s}] = get_conn(freq_pre, freq_post, is_stim);
0085:         
0086:         is_phase = endsWith(con_pre_conn{s}.label, phase_chan{:}) | startsWith(con_pre_conn{s}.label, [phase_chan{:} '-']);
0087: 
0088: 
0089:         [con_stats{s}, con_theta_stats{s}] = run_all_stats(con_pre_conn{s}, con_post_conn{s}, con_pre_conn{s}.label(is_phase));
0090:         
0091:         measure = 'pli';
0092:         stim_chans = freq_pre.label(is_stim);
0093: 
0094:     end
0095:     
0096: end
0097: 
0098: res.pre_conn = pre_conn;
0099: res.post_conn = post_conn;
...
0188:     contacts, ....
0189:     'plv', ...
0190:     [4 8]);
0191: 
0192: end
0193: 
0194: function [pre_conn, post_conn] = get_conn(freq_pre, freq_post, is_stim)
0195: 
0196: pairs = nchoosek(1:length(freq_pre.label), 2); 
0197: 
0198: pairs = pairs(any(ismember(pairs, find(is_stim)),2), :);
0199: 
0200: % remove pairs that do not include 
0201: 
0202: awPLV_pre = nan(size(pairs,1), ...
0203:     size(freq_pre.fourierspctrm,1), ...
0204:     length(freq_pre.freq));
0205: 
...
0298: pre_conn.plv = plv_pre;
0299: post_conn.plv = plv_post;
0300: 
0301: end
0302: 
0303: function [freq_pre, freq_post] = compute_fourier(dat, pre_fn, post_fn, ...
0304:     is_stim, is_phase)
0305: 
0306: % psd_pre{s}
0307: [~, freq_pre, ~] = run_rest_preproc(dat.(pre_fn), ...
0308:     dat.sr, ...
0309:     dat.contacts, ...
0310:     'all', ...
0311:     is_stim, ...
0312:     is_phase);
0313: 
0314: % psd_post{s}
0315: [~, freq_post, ~] = run_rest_preproc(dat.(post_fn), ...
0316:     dat.sr, ...
0317:     dat.contacts, ...
0318:     'all', ...
0319:     is_stim, ...
0320:     is_phase);
0321:        
0322: 
0323: end
0324: 
0325: function [awPLV, pli, dwpli, coh, icoh, plv, ciplv] = get_trial_conn(dat1, dat2)
0326: % compute resultant vector length for phase differences over time, within a
```

## `TSS-main/tss/plotting/plot_example_subj.m`

```matlab
0004: % Inputs:
0005: %   subject: subject ID
0006: %   is_control: whether the subject is a control
0007: %   ax: axes to plot on
0008: 
0009: if ~is_control
0010:     read_ep = false;
0011:     read_rest = false;
0012:     read_cl = true;
0013:     read_cl_control = false;
0014: else
0015:     read_ep = false;
0016:     read_rest = false;
0017:     read_cl = false;
0018:     read_cl_control = true;
0019: end
0020: 
0021: % directory that has processed data from GED
0022: proc_dir = '/path/to/processed/data';
0023: 
0024: [clean_dat, ged_res, dat_lf] = get_cleaned_dat(subject, ...
0025:                                                read_ep, ...
0026:                                                read_rest, ...
0027:                                                read_cl, ...
0028:                                                read_cl_control, ...
0029:                                                is_control, ...
0030:                                                proc_dir);
0031: 
0032: phase_chan = get_phase_chan(subject, clean_dat.label);
0033: 
0034: k_thr = 5; %5 kurtosis for trial exclusion, no trial exclusion
0035: z_thr = 5; %5 z score for trial exclusion,  no trial exclusion
0036: a_thr = 500;
0037: lp_flag = false;
0038: 
0039: [res, ~] = run_cl_ep_preproc(dat_lf, ...
0040:                              phase_chan, ...
0041:                              k_thr, ...
0042:                              z_thr, ...
0043:                              a_thr, ...
0044:                              lp_flag);
0045: 
0046: p2p_early = compute_p2p(res, [.015 .05]);
...
0057: writetable(dec_sep_table,['source_' subject '_dec_sep_' control_str '_table.csv']);
0058: writetable(dec_p2p_table,['source_' subject '_dec_p2p_' control_str '_table.csv']);
0059: 
0060: 
0061: end
0062: 
0063: function [clean_dat, ged_res, dat_lf] = get_cleaned_dat(subject, read_ep, read_rest, read_cl, ...
0064:     read_cl_control, is_control, proc_dir)
0065: % GET_CLEANED_DAT gets cleaned data for a subject
0066: %
0067: % Inputs:
0068: %   subject: subject ID
0069: %   read_ep: whether to read in EP data
0070: %   read_rest: whether to read in rest data
0071: %   read_cl: whether to read in closed-loop data
0072: %   read_cl_control: whether to read in closed-loop control data
0073: %   is_control: whether the data is from the control condition
0074: %
0075: 
0076: if ~exist('is_control','var')
0077:     is_control = false;
0078: end
0079: 
...
0082: else
0083:     fname = 'ged_cl_dat.mat';
0084: end
0085: 
0086: if exist([proc_dir filesep subject filesep fname],'file')
0087:     if ~is_control
0088:         load([proc_dir filesep subject filesep fname], ...
0089:             'clean_dat','ged_res','dat_lf');
0090:     else
0091:         load([proc_dir filesep subject filesep fname], ...
0092:             'clean_control_dat','ged_control_res','dat_control_lf');
0093:     end
0094: else
0095: 
0096:     [csc_dir, img_dir] = get_dirs(subject);
0097: 
0098:     if is_control
0099:         read_cl_control = true;
0100:         read_cl = false;
0101:     end
0102: 
0103:     if ~is_control
0104:         dat = get_subj_eeg(subject, ...
0105:             csc_dir, ...
0106:             img_dir, ...
0107:             read_ep, ...
0108:             read_rest, ...
0109:             read_cl, ...
0110:             read_cl_control, ...
0111:             'all');
0112:     else
0113:         dat_control = get_subj_eeg(subject, ...
0114:             csc_dir, ...
0115:             img_dir, ...
0116:             read_ep, ...
0117:             read_rest, ...
0118:             read_cl, ...
0119:             read_cl_control, ...
0120:             'all');
0121:     end
0122: 
0123:     if ~is_control
0124:         dat_lf = run_line_filter(dat.cl__epoched, ...
0125:             dat.contacts, dat.sr);
0126:     else
```

## `TSS-main/tss/plotting/plot_fig1_cl_performance.m`

```matlab
0001: function plot_fig1_cl_performance()
0002: % PLOT_FIG1_CL_PERFORMANCE plots the closed-loop performance data for Figure 1
0003: 
0004: % run or load results
0005: res = run_group_bosc();
0006: [polar_res, boot_res] = run_group_cl_phase();
0007: 
0008: % and do all of the plotting
0009: figure;
0010: 
0011: ax1 = subplot(2,6,[1:2 7:8]);
```

## `TSS-main/tss/plotting/plot_subj_cl_performance.m`

```matlab
0004: %
0005: % Input:
0006: %   subjid: subject ID
0007: %   ax1: axis for subject performance
0008: %   ax2: axis for control performance
0009: 
0010: read_ep = false;
0011: read_rest = false;
0012: read_cl = true;
0013: read_cl_control = false;
0014: phase_only = true;
0015: 
0016: [csc_dir, img_dir] = get_dirs(subjid);
0017: 
0018: if strcmp(subjid,'p20') || startsWith(subjid,'UC')
0019:     has_cl = false;
0020: else
...
0023: 
0024: if has_cl
0025: 
0026:     dat = get_subj_eeg(subjid, ...
0027:         csc_dir, ...
0028:         img_dir, ...
0029:         read_ep, ...
0030:         read_rest, ...
0031:         read_cl, ...
0032:         read_cl_control, ...
0033:         phase_only);
0034: 
0035:     dat = run_cl_preproc(dat.cl__epoched, ...
0036:         dat.sr, dat.contacts);
0037: end
0038: 
0039: % if subject has phase blind condition, show what the same phase
0040: % distribution looks like for that subject
0041: 
0042: if strcmp(subjid,'p19') || strcmp(subjid,'p17') || strcmp(subjid,'p20') || ...
0043:         startsWith(subjid,'UC')
0044: 
0045:     read_cl_control = true;
0046:     read_cl = false;
0047:     dat_control = get_subj_eeg(subjid, ...
0048:         csc_dir, ...
0049:         img_dir, ...
0050:         read_ep, ...
0051:         read_rest, ...
0052:         read_cl, ...
0053:         read_cl_control, ...
0054:         phase_only);
0055: 
0056:     dat_control = run_cl_preproc(dat_control.cl_control__epoched, ...
0057:         dat_control.sr, dat_control.contacts);
0058: 
0059:     read_cl = true;
0060:     read_cl_control = false;
0061: end
0062: 
0063: if has_cl
0064:     [est_phase, pred_dat] = pred_stim_phase(dat, 80);
0065:     % compute resultant angle length, can compute other metrics later
0066:     n = size(est_phase,2);           % number of trials
0067:     rfn = @(x) abs(sum(exp(1i*x)))/n;
```

## `TSS-main/tss/plotting/plot_theta_effects.m`

```matlab
0059: ht.FontSize = 7;
0060: 
0061: set(gca,'XTickLabel',(get(gca,'XTick')+.5)*100,'YLim',[3 8],...
0062:     'FontSize',7,'FontName','Arial');
0063: 
0064: ylabel('Power (log \muV^2)')
0065: xlabel('Stimulation Pulse (% of Total)');
0066: 
0067: axes(ax2);
0068: 
0069: % phase-synchronized condition
0070: x = power_table.Time(power_table.Subject == S & ...
0071:                      power_table.Frequency == F & ...
0072:                      power_table.Condition == categorical(2));
...
0105: ht = text(-.05 , 3.75, ['p = ' sprintf('%0.3f',pval)]);
0106: ht.FontSize = 7;
0107: 
0108: set(gca,'XTickLabel',(get(gca,'XTick')+.5)*100,'YLim',[3 8],...
0109:     'FontSize',7,'FontName','Arial');
0110: ylabel('Power (log \muV^2)')
0111: xlabel('Stimulation Pulse (% of Total)');
0112: 
0113: end
```

## `TSS-main/tss/preprocessing/run_bosc_preproc.m`

```matlab
0031: %
0032: 
0033: cfg = [];
0034: cfg.lpfilter = 'yes';
0035: cfg.lpfreq = 55;
0036: cfg.lpfiltord = 10;
0037: datout = ft_preprocessing(cfg, datin);
0038: 
0039: cfg = [];
0040: cfg.length = 8;
0041: cfg.overlap = 0.8;
0042: 
0043: datout = ft_redefinetrial(cfg, datout);
0044: 
0045: end
0046: 
0047: function dat = epoched2dat(epoched_dat, contacts, chansel, sr)
0048: 
0049: dat.dimord = 'chan_rpt_time';
0050: dat.label = contacts;
```

## `TSS-main/tss/preprocessing/run_cl_preproc.m`

```matlab
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
...
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
```

## `TSS-main/tss/preprocessing/run_ep_preproc.m`

```matlab
0005: % Input:
0006: %   pre_epoched_dat: pre-stimulation epoched data
0007: %   post_epoched_dat: post-stimulation epoched data
0008: %   fs: sampling rate
0009: %   contacts: contact names
0010: %   chansel: channel selection
0011: %   kthr: kurtosis threshold
0012: %   zthr: z-score threshold
0013: %   athr: amplitude threshold
0014: %   lp_flag: whether to low-pass filter
0015: %
0016: % Output:
0017: %   timelock_pre: timelock data for pre-stimulation data
0018: %   timelock_post: timelock data for post-stimulation data
0019: %   trial_idx_pre: trial indices for pre-stimulation data
0020: %   trial_idx_post: trial indices for post-stimulation data
...
0089: cfg = [];
0090: cfg.dftfilter = 'yes';
0091: cfg.dftfreq = [60 120 180];
0092: cfg.dftbandwidth = [.5 .5 .5];
0093: cfg.dftneighbourwidth = [1.5 1.5 1.5];
0094: 
0095: datout = ft_preprocessing(cfg, datout);
0096: 
0097: if lp_flag
0098: 
0099: cfg = [];
0100: cfg.lpfilter = 'yes';
0101: cfg.lpfreq = 80;
0102: cfg.lpfiltord = 10;
0103: % 
0104: datout = ft_preprocessing(cfg, datout);
0105: 
0106: end
0107: 
0108: end
0109: 
0110: function dat = epoched2dat(epoched_dat, contacts, chansel, sr)
0111: % EPOCHED2DAT converts epoched data to fieldtrip data structure
```

## `TSS-main/tss/preprocessing/run_line_filter.m`

```matlab
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
```

## `TSS-main/tss/preprocessing/run_rest_preproc.m`

```matlab
0001: function [dat, freq, psd] = run_rest_preproc(dat, ...
0002:     fs, contacts, chansel, is_stim, is_phase)
0003: %RUN_REST_PREPROC runs preprocessing for resting state data
0004: %
0005: % Input:
0006: %   dat: epoched data
0007: %   fs: sampling rate
0008: %   contacts: contact names
0009: %   chansel: channel selection
0010: %   is_stim: whether channel is a stimulation channel
0011: %   is_phase: whether channel is a phase channel
0012: %
0013: % Output:
0014: %   dat: epoched data
0015: %   freq: frequency data
0016: %   psd: power spectral density data
0017: %
...
0021: 
0022: % old settings here
0023: cfg = [];
0024: cfg.length = 8;
0025: cfg.overlap = 0.8;
0026: 
0027: dat = ft_redefinetrial(cfg, dat);
0028: 
0029: % exclude via kurtosis thresh
0030: kthr = 4.5;
0031: k = squeeze(kurtosis(cat(3, dat.trial{:}),[],2));
0032: cfg = [];
0033: cfg.trials = all(k(is_stim | is_phase,:) < kthr); %mean(k<kthr) > .95; %all(k < kthr);
0034: 
0035: if ~any(cfg.trials)
0036:     cfg.trials = true(size(cfg.trials));
0037:     warning('No trials were selected. All trials will be used.');
0038: end
0039: 
0040: dat = ft_selectdata(cfg, dat);
...
0081: 
0082: cfg.dftfreq = dft_frequencies;
0083: cfg.dftbandwidth = repmat(0.5, size(dft_frequencies)); % bandwidth for each frequency
0084: cfg.dftneighbourwidth = [1.5 1.5 1.5];
0085: cfg.dftreplace = 'neighbour';
0086: 
0087: datout = ft_preprocessing(cfg, datin);
0088: 
0089: end
0090: 
0091: function dat = epoched2dat(epoched_dat, contacts, chansel, sr)
0092: % EPOCHED2DAT converts epoched data to continuous data format
0093: %
0094: % Input:
```

## `TSS-main/tss/utils/get_cl_control_phase_chan.m`

```matlab
0007: %
0008: % Output:
0009: %   phase_chan: phase channel name
0010: %
0011: 
0012: % stim and phase lock info
0013: stim = get_stim_info(subjid);
0014: 
0015: if contains(stim.cl_control_phase,'-')
0016:     stim.cl_control_phase = strsplit(stim.cl_control_phase,'-');
0017: end
0018: 
0019: is_phase = false(size(contacts));
0020: for c = 1:length(contacts)
```

## `TSS-main/tss/utils/get_epoched_eeg.m`

```matlab
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
...
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
...
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
```

## `TSS-main/tss/utils/get_group_synchrony.m`

```matlab
0025:     end
0026: 
0027:     if isempty(pre_conn{s})
0028:         continue
0029:     end
0030: 
0031:     stim_info = get_stim_info(subjects{s});
0032: 
0033:     is_phase = endsWith(pre_conn{s}.label, stim_info.phase) | ...
0034:                startsWith(pre_conn{s}.label, [stim_info.phase '-']);
0035: 
0036:     % pre
0037:     tmp = mean(pre_conn{s}.(measure)(is_phase,:,:), 'omitnan'); % across both stim contacts
0038:     tmp = squeeze(tmp);
```

## `TSS-main/tss/utils/get_phase_chan.m`

```matlab
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
```

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
...
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
...
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
...
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

## `TSS-main/tss/utils/robust_ap_fit.m`

```matlab
0002: % ROBUST_AP_FIT robust aperiodic fit to replace BOSC_bgfit
0003: %
0004: % Inputs:
0005: %   freqs - frequency values
0006: %   ps - power values
0007: %   ap_guess - initial guess for aperiodic fit
0008: %   ap_amp_thresh - amplitude threshold for aperiodic fit
0009: %
0010: % Outputs:
0011: %   ap_params - parameters for aperiodic fit
0012: %   ap_ps - aperiodic fit
0013: %
0014: 
0015: % ps should be in log10 scale, as is the ap_ps output
...
0024: init_ps = gen_ap(init_params, freqs);
0025: 
0026: % flatten power spectrum from initial
0027: flat_ps = ps - init_ps;
0028: flat_ps(flat_ps<0)=0;
0029: 
0030: % amplitude threshold
0031: p_thresh = prctile(flat_ps, ap_amp_thresh);
0032: mask = flat_ps <= p_thresh;
0033: freqs_use= freqs(mask); % frequencies in aperiodic range
0034: ps_use = ps(mask);
0035: 
0036: % find the function to use
0037: if length(init_params) == 2
```

%requires SPM to be installed and in the path
clear all
rng shuffle


for ttt = 1:5

switch ttt
    case 1
trials = 25;     
    case 2
trials = 50;     
    case 3
trials = 100;     
    case 4
trials = 200;     
    case 5
trials = 400;     
end 

runs = 5000;

trials1 = trials*7;
ts = zeros(runs, trials1);
ts_est = zeros(runs, trials1);
dPE_ts_est = zeros(runs, trials1);
dPE2_ts_est = zeros(runs, trials1);
dPE3_ts_est = zeros(runs, trials1);
winloss_ts = zeros(runs, trials1);
tsm_est = zeros(runs, trials1);
output = zeros(runs,5);

prec = 0.1; %05;

TR = 2;

HRF = spm_hrf(TR);

mdl = arima(2,0,0);
ts_len = trials1+16;

for i = 1:runs

% variation in ground truth alpha and lambda
%true alpha and lambda are assumed to be related to e.g. anhedonia
lambda(i) = rand(1)*0.5 + 0.75; 
alpha(i) = rand(1)*0.5 + 0.2; 
hrf_scale(i) = rand(1) + 0.5;

lambda_est(i) = 1;
alpha_est(i) = 0.45;
alpha_est_h(i) = 0.7;
alpha_est_l(i) = 0.2;




%generate reinforcement contingencies
    % 50% contingency with slow drift

drift(i) = rand(1)*0.4; 
pR(i,1) = 0.5;
QA(i,1) = 0.5;
QA_est(i,1) = 0.5;
%QA_est_h(i,1) = 0.5;
%QA_est_l(i,1) = 0.5;

s_part = 2+(rand(1)*2);
snr(i) = s_part/(s_part + 1);

for t = 1:trials

if t>1    
pR(i,t) = pR(i,t-1) + randn(1)*drift(1);
if pR(i,t)>1
pR(i,t) = 1;
end
if pR(i,t)<0
pR(i,t) = 0;
end
end

pR1 = rand(1);

if pR1<pR(i,t)
reward(i,t) = 1;
winloss_ts(i, 1+((t-1)*7)) = 1;
else
reward(i,t) = 0;
winloss_ts(i, 1+((t-1)*7)) = -1;
end

PE(i,t) = (lambda(i)*reward(i,t)) - QA(i,t);
QA(i,t+1) = QA(i,t) + alpha(i)*PE(i,t);

PE_est(i,t) = (lambda_est(i)*reward(i,t)) - QA_est(i,t);
QA_est(i,t+1) = QA_est(i,t) + alpha_est(i)*PE_est(i,t);

% PE_est_h(i,t) = (lambda_est(i)*reward(i,t)) - QA_est_h(i,t);
% QA_est_h(i,t+1) = QA_est_h(i,t) + alpha_est_h(i)*PE_est_h(i,t);
% 
% PE_est_l(i,t) = (lambda_est(i)*reward(i,t)) - QA_est_l(i,t);
% QA_est_l(i,t+1) = QA_est_l(i,t) + alpha_est_l(i)*PE_est_l(i,t);


% dPE_est(i,t) = PE_est_h(i,t) - PE_est_l(i,t);
% PEm_est(i,t) = (PE_est_h(i,t) + PE_est_l(i,t))/2;

%  if t==1
%  dPE(i,t) = 0;
%  dPE2_est(i,t) = 0;
%  else
%  dPE(i,t) = PE(i,t) - PE(i,t-1);    
%  dPE2_est(i,t) = PE_est(i,t) - PE_est(i,t-1);
%  end
% 

ts(i, 1+((t-1)*7)) = PE(i,t);

ts_est(i, 1+((t-1)*7)) = PE_est(i,t);
% tsm_est(i, 1+((t-1)*7)) = PEm_est(i,t);
% dPE_ts_est(i, 1+((t-1)*7)) = dPE_est(i,t);
% dPE2_ts_est(i, 1+((t-1)*7)) = dPE2_est(i,t);
%     
end

dPE3_est(i,:) = gradient(PE_est(i,:));

for t = 1:trials

dPE3_ts_est(i, 1+((t-1)*7)) = dPE3_est(i,t);    
    
end

%convolve ts + add noise

ts_conv(i,:) = conv(ts(i,:), HRF*hrf_scale(i));

noise_alpha(i) = 0.8 + (rand(1)*0.4);
dcn = dsp.ColoredNoise(noise_alpha(i),ts_len,1);
noise_out = step(dcn);
noise_out = (noise_out-(mean(noise_out)))/(std(noise_out));

fMRI_ts(i,:) = snr(i)*ts_conv(i,:) + (1-snr(i))*noise_out';



%convolve designs
ts_z(i,:) = (ts_est(i,:)-mean(ts_est(i,:)))/(std(ts_est(i,:)));
deriv_z(i,:) = (dPE3_ts_est(i,:)-mean(dPE3_ts_est(i,:)))/(std(dPE3_ts_est(i,:)));

ts_est_conv(i,:) = conv(ts_z(i,:), HRF);
deriv_est_conv(i,:) = conv(deriv_z(i,:), HRF);
for lin = 1:ts_len
linear(lin) = lin;
end

X_DM = ([ts_est_conv(i,:)' deriv_est_conv(i,:)' linear']);

%EstMdl = estimate(mdl, fMRI_ts(i,1:(ts_len-2))', 'X', X_DM);

[EstMdl, ~, LogL] = estimate(mdl, fMRI_ts(i,1:(ts_len-2))', 'X', X_DM);
bb = EstMdl.Beta;
output(i,1) = bb(1);
output(i,2) = bb(2);
output(i,3) = bb(3);
[~,bic] = aicbic(LogL, (2+3+1), (ts_len-2));
output(i,4) = bic;


%output(i,4) = (bb(1)*(bb(1)^2)) + ((bb(2)^2)*EstMdl.Variance);
%output(i,4) = bb(1)*(((bb(1)^2)+ (bb(2)^2))^0.5);

 X_DM1 = ([ts_est_conv(i,:)' linear']);
 
[EstMdl1, ~, LogL1] =  estimate(mdl, fMRI_ts(i,1:(ts_len-2))', 'X', X_DM1);
[~,bic1] = aicbic(LogL1, (2+2+1), (ts_len-2));
output(i,5) = bic1;

if bic<bic1
    output(i,6) = 1;
else
    output(i,6) = 0;
end

[rrr] = corrcoef(ts_est_conv(i,:), deriv_est_conv(i,:));
output(i,7) = rrr(2);


% bb1 = EstMdl1.Beta;
% output(i,5) = bb1(1);
% output(i,6) = bb1(2);
% 
% 
% hybrid_X = zscore((output(i,1)*X_DM(:,1)))+((output(i,2)*X_DM(:,2)));
% 
% X_DM2 = ([hybrid_X linear']);
% 
% EstMdl2 = estimate(mdl, fMRI_ts(i,1:(ts_len-2))', 'X', X_DM2);
% bb2 = EstMdl2.Beta;
% output(i,7) = bb2(1);
% output(i,8) = bb2(2);


end


stats_final1 = regstats(output(:,1), [lambda' alpha' drift' snr' noise_alpha']);
stats_final2 = regstats(output(:,2), [lambda' alpha' drift' snr' noise_alpha']);
stats_final3 = regstats(output(:,3), [lambda' alpha' drift' snr' noise_alpha']);
stats_final4 = regstats(output(:,4), [lambda' alpha' drift' snr' noise_alpha']);
stats_final5 = regstats(output(:,1), [lambda' alpha' hrf_scale' drift' snr' noise_alpha']);
stats_final6 = regstats(output(:,2), [lambda' alpha' hrf_scale' drift' snr' noise_alpha']);
stats_final7 = regstats(output(:,3), [lambda' alpha' hrf_scale' drift' snr' noise_alpha']);
stats_final8 = regstats(output(:,4), [lambda' alpha' hrf_scale' drift' snr' noise_alpha']);
stats_final9 = regstats((output(:,4) - output(:,5)), [lambda' alpha' hrf_scale' drift' snr' noise_alpha'], 'purequadratic');

%MnrMDL = fitmnr(output(:,6), [zscore(lambda)' zscore(abs(alpha-0.45))' zscore(hrf_scale)' zscore(drift)' zscore(snr)' zscore(noise_alpha)' zscore(abs(alpha-0.45))'.*zscore(noise_alpha)' zscore(abs(alpha-0.45))'.*zscore(snr)']);

[log_betas, dev, log_stats]  = mnrfit([zscore(lambda)' zscore(abs(alpha-0.45))' zscore(hrf_scale)' zscore(drift)' zscore(snr)' zscore(noise_alpha)' zscore(abs(alpha-0.45))'.*zscore(noise_alpha)' zscore(abs(alpha-0.45))'.*zscore(snr)'], categorical(output(:,6)));

% stats_final5 = regstats(output(:,5), [lambda' alpha' drift' snr' noise_alpha']);
% stats_final6 = regstats(output(:,6), [lambda' alpha' drift' snr' noise_alpha']);
% stats_final7 = regstats(output(:,7), [lambda' alpha' drift' snr' noise_alpha']);
% stats_final8 = regstats(output(:,8), [lambda' alpha' drift' snr' noise_alpha']);

stats_final_a = regstats(alpha', [output(:,1) output(:,2)]);
stats_final_l = regstats(lambda', [output(:,1) output(:,2)]);
stats_final_ar = regstats(alpha', [output(:,2)]);
stats_final_lr = regstats(lambda', [output(:,1)]);
stats_final_lr2 = regstats(lambda', [output(:,4)]);

stats_hold{1, ttt} = stats_final1;
stats_hold{2, ttt} = stats_final2;
stats_hold{3, ttt} = stats_final3;
stats_hold{4, ttt} = stats_final4;
 stats_hold{5, ttt} = stats_final5;
 stats_hold{6, ttt} = stats_final6;
 stats_hold{7, ttt} = stats_final7;
 stats_hold{8, ttt} = stats_final8;
stats_hold{9, ttt} = stats_final_a;
stats_hold{10, ttt} = stats_final_l;
stats_hold{11, ttt} = stats_final_ar;
stats_hold{12, ttt} = stats_final_lr;
stats_hold{13, ttt} = stats_final_lr2;
stats_hold{14, ttt} = stats_final9;
stats_hold{15, ttt} = log_stats;

stats_hold2{1, ttt} = mean(output(:,7));
stats_hold2{2, ttt} = std(output(:,7));
stats_hold2{3, ttt} = (sum(output(:,6)))/runs;

%save('sim_output_ES_update.mat', 'stats_hold', '-v7.3')

ccc = corrcoef(output(:,1), lambda);
corr_output(1, ttt) = ccc(2);
ccc = corrcoef(output(:,1), alpha);
corr_output(2, ttt) = ccc(2);
ccc = corrcoef(output(:,2), lambda);
corr_output(3, ttt) = ccc(2);
ccc = corrcoef(output(:,2), alpha);
corr_output(4, ttt) = ccc(2);
ccc = corrcoef(alpha, lambda);
corr_output(5, ttt) = ccc(2);
pcc = partialcorr(output(:,1), lambda', hrf_scale');
corr_output(6, ttt) = pcc;
pcc = partialcorr(output(:,2), alpha', hrf_scale');
corr_output(7, ttt) = pcc;

%save('sim_output_ES_update1.mat', 'corr_output', '-v7.3')

clear dPE2_est dPE3_est dPE_est ts_conv fMRI_ts ts_z deriv_z
clear ts_est_conv deriv_est_conv linear
clear QA QA_est PR drift lambda alpha reward

end
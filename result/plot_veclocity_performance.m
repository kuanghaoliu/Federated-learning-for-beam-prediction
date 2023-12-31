clear all;
close all;
clc

BL4 = zeros(6, 10, 9);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['./proposed/vec/V_' num2str(v) 'C_36_E10_R60_B32.mat']);
    BL4(count, :, :) = squeeze(BL_test(end, :, :));
end

BL3 = zeros(6, 10, 9);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['./benchmark/WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_ODE_3CNN_1LSTM_160epoch_v1.mat']);
    BL3(count, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
end

BL2 = zeros(6, 10, 9);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
    load(['./benchmark/WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_ICC_3CNN_1LSTM_160epoch_m=4_v1.mat']);
    BL2(count, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, 1)), 3));
end

BL1 = zeros(6, 10, 9);
count = 0;
for v = 5 : 5 : 30
    count = count + 1;
   load(['./benchmark/WCL_v' num2str(v) '_a' num2str(v * 0.2) '.0_25dBm_LSTM_3CNN_1LSTM_v1.mat']);
    BL1(count, :, :) = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));
end


load(['./benchmark/EKF_result.mat']);
BL0 = beam_loss_sery_total(2 : end, :, :);
BL0 = permute(BL0 ,[3, 2, 1]);


figure;
hold on;
box on;
grid on;
xlabel('UE velocity $v$ (m/s)', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(5 : 5 : 30, squeeze(mean(mean(BL0(:, :, :), 3), 2)),  '-*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL1(:, :, :), 3), 2)), 'm-v',  'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL2(:, :, :), 3), 2)), 'b-^', 'linewidth', 1.5);
plot(5 : 5 : 30, squeeze(mean(mean(BL3(:, :, :), 3), 2)), 'r-o', 'linewidth', 1.5);

plot(5 : 5 : 30, squeeze(mean(mean(BL4(:, :, :), 3), 2)),  '-*','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
legend('EKF','Conventional LSTM', 'Cascaded LSTM', 'ODE-LSTM', 'Proposed',...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);


figure;
hold on;
box on;
grid on;
xlabel('UE velocity $v$ (m/s)', 'interpreter', 'latex','fontsize',14);
ylabel('Sum rate (bps/Hz)', 'interpreter', 'latex','fontsize',14);
%plot(5 : 5 : 30, squeeze(mean(mean(BL0(:, :, :), 3), 2)),  '-*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(5 : 5 : 30, [4.29136 4.01149 3.78133 3.54713 3.34318 3.14296], 'm-v',  'linewidth', 1.5);
plot(5 : 5 : 30, [4.50111 4.34674 4.18344 4.01490 3.86250 3.69657], 'b-^', 'linewidth', 1.5);
plot(5 : 5 : 30, [4.54131 4.47184 4.43521 4.39956 4.40592 4.38661], 'r-o', 'linewidth', 1.5);
plot(5 : 5 : 30, [4.59652 4.58113 4.57470 4.57923 4.57895 4.54753],  '-*','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
legend('Conventional LSTM', 'Cascaded LSTM', 'ODE-LSTM', 'Proposed',...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);
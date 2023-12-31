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


% 10m/s
figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL0(2, :, :), 2)), '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL1(2, :, :), 2)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL2(2, :, :), 2)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL3(2, :, :), 2)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4(2, :, :), 2)),  '-pentagram','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
legend('EKF','Conventional LSTM', 'Cascaded LSTM', 'ODE-LSTM', ...
    'Proposed', ...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);

% 20m/s
figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL0(4, :, :), 2)), '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL1(4, :, :), 2)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL2(4, :, :), 2)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL3(4, :, :), 2)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4(4, :, :), 2)),  '-pentagram','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
legend('EKF','Conventional LSTM', 'Cascaded LSTM', 'ODE-LSTM', ...
    'Proposed', ...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);

% 30m/s
figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL0(6, :, :), 2)), '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL1(6, :, :), 2)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL2(6, :, :), 2)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL3(6, :, :), 2)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4(6, :, :), 2)),  '-pentagram','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
legend('EKF','Conventional LSTM', 'Cascaded LSTM', 'ODE-LSTM', ...
    'Proposed', ...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);
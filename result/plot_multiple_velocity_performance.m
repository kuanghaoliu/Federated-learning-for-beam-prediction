clear all;
close all;
clc


load(['./proposed/cons/Cons1_C36_E10_R50_B32.mat']);
BL4 = squeeze(BL_test(end, :, :));

load(['./proposed/cons/WCL_cons_25dBm_ODE_3CNN_1LSTM_160epoch_v1.mat']);
BL3 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));

load(['./proposed/cons/WCL_cons_25dBm_ICC_3CNN_1LSTM_160epoch_m=4_v1.mat']);
BL2 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));

load(['./proposed/cons/WCL_cons_25dBm_LSTM_3CNN_1LSTM_v1.mat']);
BL1 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));


figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL1, 1)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL2, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL3, 1)), 'r-o', 'linewidth', 1.5);

plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4, 1)),  '-pentagram','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
legend('Conventional LSTM', 'Cascaded LSTM', 'ODE-LSTM', ...
    'Proposed', ...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load(['./proposed/rand/Rand1_C36_E10_R50_B32.mat']);
BL8 = squeeze(BL_test(end, :, :));

load(['./proposed/rand/WCL_vrand_25dBm_ODE_3CNN_1LSTM_160epoch_v1.mat']);
BL5 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));

load(['./proposed/rand/WCL_vrand_25dBm_ICC_3CNN_1LSTM_160epoch_m=4_v1.mat']);
BL6 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));

load(['./proposed/rand/WCL_vrand_25dBm_LSTM_3CNN_1LSTM_v1.mat']);
BL7 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));


figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL5, 1)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL6, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL7, 1)), 'r-o', 'linewidth', 1.5);

plot(0.1 : 0.1 : 0.9, squeeze(mean(BL8, 1)),  '-pentagram','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
legend('Conventional LSTM', 'Cascaded LSTM', 'ODE-LSTM', ...
    'Proposed', ...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);

figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4, 1)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL8, 1)), 'b-square', 'linewidth', 1.5);
ylim([0 1])
legend('constant velocity dataset','random velocity dataset', ...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);



figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4, 1)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL8, 1)), 'b-square', 'linewidth', 1.5);
ylim([0 1])

H = axes('Position',[0.2,0.15,0.5,0.4]);
hold on;
box on;
grid on;
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4, 1)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL8, 1)), 'b-square', 'linewidth', 1.5);
xlim([0.1 0.9])
set(H);
legend('Constant velocity dataset','Random velocity dataset', ...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);
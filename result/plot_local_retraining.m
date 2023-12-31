clear all;
close all;
clc

load(['./proposed/cons/cons1_cl_E1_R10.mat']);
BL7 = squeeze(BL_test(1, :, :));

load(['./proposed/cons/cons1_cl_E1_R50_1024.mat']);
BL6 = squeeze(BL_test(1, :, :));

load(['./proposed/cons/cons1_cl_E1_R10.mat']);
BL5 = squeeze(BL_test(end, :, :));

load(['./proposed/cons/cons1_cl_E10_R50.mat']);
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
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL5, 1)),  '-v','Color',[0.4660 0.6740 0.1880], 'linewidth', 1.5);
legend('Conventional LSTM', 'Cascaded LSTM', 'ODE-LSTM', ...
    'Personalized ($R=30$)', 're-training ($R=10$)',...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);

figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4, 1)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL6, 1)), 'r--o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL5, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL7, 1)), 'b--square', 'linewidth', 1.5);
ylim([0 1])
legend('Personalized ($R=30$)', 'Proposed ($R=30$)',...
    're-training ($R=10$)', 'Proposed ($R=10$)',...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);


figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4, 1)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL6, 1)), 'r--o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL5, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL7, 1)), 'b--square', 'linewidth', 1.5);
ylim([0 1])
H = axes('Position',[0.2,0.15,0.5,0.4]); 
hold on;
box on;
grid on;
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4, 1)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL6, 1)), 'r--o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL5, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL7, 1)), 'b--square', 'linewidth', 1.5);
xlim([0.1 0.9])
set(H);
legend('With local re-training ($R=30$)', 'Without local re-training ($R=30$)',...
    'Re-training ($R=10$)', 'Proposed ($R=10$)',...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load(['./proposed/rand/rand1_cl_E1_R10.mat']);
BL14 = squeeze(BL_test(1, :, :));

load(['./proposed/rand/rand1_cl_E10_R50.mat']);
BL13 = squeeze(BL_test(1, :, :));

load(['./proposed/rand/rand1_cl_E1_R10.mat']);
BL12 = squeeze(BL_test(end, :, :));

load(['./proposed/rand/rand1_cl_E10_R50.mat']);
BL11 = squeeze(BL_test(end, :, :));

load(['./proposed/rand/WCL_vrand_25dBm_ODE_3CNN_1LSTM_160epoch_v1.mat']);
BL10 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));

load(['./proposed/rand/WCL_vrand_25dBm_ICC_3CNN_1LSTM_160epoch_m=4_v1.mat']);
BL9 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));

load(['./proposed/rand/WCL_vrand_25dBm_LSTM_3CNN_1LSTM_v1.mat']);
BL8 = squeeze(mean(squeeze(BL_eval(:, :, end, :)), 3));


figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL8, 1)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL9, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL10, 1)), 'r-o', 'linewidth', 1.5);

plot(0.1 : 0.1 : 0.9, squeeze(mean(BL11, 1)),  '-pentagram','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL12, 1)),  '-v','Color',[0.4660 0.6740 0.1880], 'linewidth', 1.5);
legend('Conventional LSTM', 'Cascaded LSTM', 'ODE-LSTM', ...
    'Personalized ($R=30$)', 're-training ($R=10$)',...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);

figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL11, 1)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL13, 1)), 'r--o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL12, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL14, 1)), 'b--square', 'linewidth', 1.5);
ylim([0 1])
legend('Personalized ($R=30$)', 'Proposed ($R=30$)',...
    're-training ($R=10$)', 'Proposed ($R=10$)',...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);

figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL11, 1)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL13, 1)), 'r--o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL12, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL14, 1)), 'b--square', 'linewidth', 1.5);
ylim([0 1])

H = axes('Position',[0.2,0.15,0.5,0.4]);
hold on;
box on;
grid on;
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL11, 1)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL13, 1)), 'r--o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL12, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL14, 1)), 'b--square', 'linewidth', 1.5);
xlim([0.1 0.9])
set(H);
legend('With local re-training ($R=30$)', 'Without local re-training ($R=30$)',...
    'With local re-training ($R=10$)', 'Without local re-training ($R=10$)',...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL11, 1)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4, 1)), 'r--o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL12, 1)), 'b-square', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL5, 1)), 'b--square', 'linewidth', 1.5);
legend('random velocity dataset ($R=30$)', 'constant velocity dataset ($R=30$)',...
    'random velocity dataset ($R=10$)', 'constant velocity dataset ($R=10$)',...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);
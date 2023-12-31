clear all;
close all;
clc

load(['./round/mix/mix_1.0.mat']);
BL1 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/mix/mix_2.0.mat']);
BL2 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/mix/mix_3.0.mat']);
BL3 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/mix/mix_4.0.mat']);
BL4 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/mix/mix_5.0.mat']);
BL5 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/mix/mix_6.0.mat']);
BL6 = mean(squeeze(BL_test(end, :, :)),"all");

BL_constant = [BL1 BL2 BL3 BL4 BL5 BL6]

load(['./round/rand/rand_1.0.mat']);
BL1 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/rand/rand_2.0.mat']);
BL2 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/rand/rand_3.0.mat']);
BL3 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/rand/rand_4.0.mat']);
BL4 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/rand/rand_5.0.mat']);
BL5 = mean(squeeze(BL_test(end, :, :)),"all");

load(['./round/rand/rand_6.0.mat']);
BL6 = mean(squeeze(BL_test(end, :, :)),"all");

BL_rand = [BL1 BL2 BL3 BL4 BL5 BL6]

figure;
hold on;
box on;
grid on;
xlabel('Communication round $R$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(5 : 5 : 30, BL_constant,  '-*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(5 : 5 : 30, BL_rand, 'm-v',  'linewidth', 1.5);

legend('Constant velocity dataset', 'Random velocity dataset', ...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);
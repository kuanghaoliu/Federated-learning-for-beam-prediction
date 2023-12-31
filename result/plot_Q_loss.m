clear all;
close all;
clc

load(['./proposed/Q/Rand/Rand_Q13_C36_E10_R50_B32.mat']);
RBL_13 = squeeze(BL_test(30, :, :));
Rand_13 = squeeze(loss_test(30));
LOSS13 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Rand/Rand_Q11_C36_E10_R50_B32.mat']);
RBL_11 = squeeze(BL_test(30, :, :));
Rand_11 = squeeze(loss_test(30));
LOSS11 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Rand/Rand_Q9_C36_E10_R50_B32.mat']);
RBL_9 = squeeze(BL_test(30, :, :));
Rand_9 = squeeze(loss_test(30));
LOSS9 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Rand/Rand_Q7_C36_E10_R50_B32.mat']);
RBL_7 = squeeze(BL_test(30, :, :));
Rand_7 = squeeze(loss_test(30));
LOSS7 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Rand/Rand_Q5_C36_E10_R50_B32.mat']);
RBL_5 = squeeze(BL_test(30, :, :));
Rand_5 = squeeze(loss_test(30));
LOSS5 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Rand/Rand_Q3_C36_E10_R50_B32.mat']);
RBL_3 = squeeze(BL_test(30, :, :));
Rand_3 = squeeze(loss_test(30));
LOSS3 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Rand/Rand_Q1_C36_E10_R50_B32.mat']);
RBL_1 = squeeze(BL_test(30, :, :));
Rand_1 = squeeze(loss_test(30));
LOSS1 = squeeze(loss_test(:,1:30));

figure;
hold on;
box on;
grid on;
xlabel('Comunication Round (random)', 'interpreter', 'latex');
ylabel('Cross-Entropy Loss');
plot(LOSS13/126, 'r-x', 'linewidth', 1.5);
plot(LOSS11/108, 'g-o', 'linewidth', 1.5);
plot(LOSS9/90, 'b->', 'linewidth', 1.5);
plot(LOSS7/72, 'c-<', 'linewidth', 1.5);
plot(LOSS5/54, 'm-^', 'linewidth', 1.5);
plot(LOSS3/36, 'y-+', 'linewidth', 1.5);
%plot(LOSS1/18, '-p','Color',[1    0.6   0.07], 'linewidth', 1.5);

legend('$Q=13$', '$Q=11$', '$Q=9$', '$Q=7$', '$Q=5$', '$Q=3$',...
    'interpreter', 'latex',...
    'Location','northeast');

figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(RBL_13, 1)),  '-*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(RBL_11, 1)), 'm-v',  'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(RBL_9, 1)), 'b-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(RBL_7, 1)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(RBL_5, 1)),  '-+','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(RBL_3, 1)), 'c-<', 'linewidth', 1.5);
legend('$Q=13$', '$Q=11$', '$Q=9$', '$Q=7$', '$Q=5$', 'Q=3',...
    'Proposed', ...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load(['./proposed/Q/Constant/Cons_Q13_C36_E10_R50_B32.mat']);
CBL_13 = squeeze(BL_test(30, :, :));
Constant_13 = squeeze(loss_test(30));
LOSS13 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Constant/Cons_Q11_C36_E10_R50_B32.mat']);
CBL_11 = squeeze(BL_test(30, :, :));
Constant_11 = squeeze(loss_test(30));
LOSS11 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Constant/Cons_Q9_C36_E10_R50_B32.mat']);
CBL_9 = squeeze(BL_test(30, :, :));
Constant_9 = squeeze(loss_test(30));
LOSS9 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Constant/Cons_Q7_C36_E10_R50_B32.mat']);
CBL_7 = squeeze(BL_test(30, :, :));
Constant_7 = squeeze(loss_test(30));
LOSS7 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Constant/Cons_Q5_C36_E10_R50_B32.mat']);
CBL_5 = squeeze(BL_test(30, :, :));
Constant_5 = squeeze(loss_test(30));
LOSS5 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Constant/Cons_Q3_C36_E10_R50_B32.mat']);
CBL_3 = squeeze(BL_test(30, :, :));
Constant_3 = squeeze(loss_test(30));
LOSS3 = squeeze(loss_test(:,1:30));

load(['./proposed/Q/Constant/Cons_Q1_C36_E10_R50_B32.mat']);
CBL_1 = squeeze(BL_test(30, :, :));
Constant_1 = squeeze(loss_test(30));
LOSS1 = squeeze(loss_test(:,1:30));

figure;
hold on;
box on;
grid on;
xlabel('Comunication Round (constant)', 'interpreter', 'latex');
ylabel('Cross-Entropy Loss');
plot(LOSS13/126, 'r-x', 'linewidth', 1.5);
plot(LOSS11/108, 'g-o', 'linewidth', 1.5);
plot(LOSS9/90, 'b->', 'linewidth', 1.5);
plot(LOSS7/72, 'c-<', 'linewidth', 1.5);
plot(LOSS5/54, 'm-^', 'linewidth', 1.5);
plot(LOSS3/36, 'y-+', 'linewidth', 1.5);
%plot(LOSS1/18, '-p','Color',[1    0.6   0.07], 'linewidth', 1.5);

legend('$Q=13$', '$Q=11$', '$Q=9$', '$Q=7$', '$Q=5$', '$Q=3$',...
    'interpreter', 'latex',...
    'Location','northeast');

figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(CBL_13, 1)),  '-*','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(CBL_11, 1)), 'm-v',  'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(CBL_9, 1)), 'b-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(CBL_7, 1)), 'r-o', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(CBL_5, 1)),  '-+','Color',[0.4940 0.1840 0.5560], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(CBL_3, 1)), 'c-<', 'linewidth', 1.5);
legend('$Q=13$', '$Q=11$', '$Q=9$', '$Q=7$', '$Q=5$', 'Q=3',...
    'Proposed', ...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Rand = [mean(Rand_1/18, 'all'), mean(Rand_3/36, 'all'), mean(Rand_5/54, 'all'),...
    mean(Rand_7/72, 'all'), mean(Rand_9/90, 'all'), mean(Rand_11/108, 'all'), mean(Rand_13/126, 'all')];
Mix = [mean(Constant_1/18, 'all'), mean(Constant_3/36, 'all'), mean(Constant_5/54, 'all'),...
    mean(Constant_7/72, 'all'), mean(Constant_9/90, 'all'), mean(Constant_11/108, 'all'), mean(Constant_13/126, 'all')];

figure;
hold on;
box on;
grid on;
xlabel('Number of previous beam training $Q$', 'interpreter', 'latex','fontsize',14);
ylabel('Cross-Entropy Loss', 'interpreter', 'latex','fontsize',14);
plot([1 3 5 7 9 11 13], Rand, '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot([1 3 5 7 9 11 13], Mix, 'm-^', 'linewidth', 1.5);
legend('random velocity dataset', ...
    'constant velocity dataset', ...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);

Rand_BL = [mean(RBL_1, 'all'), mean(RBL_3, 'all'), mean(RBL_5, 'all'),...
    mean(RBL_7, 'all'), mean(RBL_9, 'all'), mean(RBL_11, 'all'), mean(RBL_13, 'all')];
Mix_BL = [mean(CBL_1, 'all'), mean(CBL_3, 'all'), mean(CBL_5, 'all'),...
    mean(CBL_7, 'all'), mean(CBL_9, 'all'), mean(CBL_11, 'all'), mean(CBL_13, 'all')];
figure;
hold on;
box on;
grid on;
xlabel('Number of previous beam training $Q$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot([1 3 5 7 9 11 13], Rand_BL, '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot([1 3 5 7 9 11 13], Mix_BL, 'm-^', 'linewidth', 1.5);
legend('random velocity dataset', ...
    'constant velocity dataset', ...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',12);
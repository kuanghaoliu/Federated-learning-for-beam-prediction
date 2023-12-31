clear all;
close all;
clc


load(['./proposed/cons/Cons1_C36_E10_R50_B32.mat']);
BL3 = squeeze(BL_test(end, :, :));
B3 = squeeze(mean(BL_test(1:30, :, :),[2,3]));
ACC3 = squeeze(acur_test(:,1:30));
LOSS3 = squeeze(loss_test(:,1:30));

load(['./proposed/cons/Cons2_C18_E10_R50_B32.mat']);
BL2 = squeeze(BL_test(30, :, :));
B2 = squeeze(mean(BL_test(1:30, :, :),[2,3]));
ACC2 = squeeze(acur_test(:,1:30));
LOSS2 = squeeze(loss_test(:,1:30));

load(['./proposed/cons/Cons4_C9_E10_R50_B32.mat']);
BL1 = squeeze(BL_test(30, :, :));
B1 = squeeze(mean(BL_test(1:30, :, :),[2,3]));
ACC1 = squeeze(acur_test(:,1:30));
LOSS1 = squeeze(loss_test(:,1:30));

figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL3, 1)), '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL2, 1)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL1, 1)), 'b-square', 'linewidth', 1.5);

legend('$K=36$', '$K=18$',...
    '$K=9$',...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);

figure;
hold on;
box on;
grid on;
xlabel('Comunication round', 'interpreter', 'latex','fontsize',16);
ylabel('Accuracy', 'interpreter', 'latex','fontsize',16);
plot(ACC3, '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(ACC2, 'm-^', 'linewidth', 1.5);
plot(ACC1, 'b-square', 'linewidth', 1.5);

legend('$K=36$', '$K=18$',...
    '$K=9$',...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',16);


figure;
hold on;
box on;
grid on;
xlabel('Comunication round', 'interpreter', 'latex','fontsize',16);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',16);
plot(B3, '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(B2, 'm-^', 'linewidth', 1.5);
plot(B1, 'b-square', 'linewidth', 1.5);

legend('$K=36$', '$K=18$',...
    '$K=9$',...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',16);


figure;
hold on;
box on;
grid on;
xlabel('Comunication round', 'interpreter', 'latex','fontsize',16);
ylabel('Cross-Entropy Loss','fontsize',16);
plot(LOSS3, '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(LOSS2, 'm-^', 'linewidth', 1.5);
plot(LOSS1, 'b-square', 'linewidth', 1.5);

legend('$K=36$', '$K=18$',...
    '$K=9$',...
    'interpreter', 'latex',...
    'Location','northeast','fontsize',16);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load(['./proposed/rand/Rand1_C36_E10_R50_B32.mat']);
BL6 = squeeze(BL_test(30, :, :));
B6 = squeeze(mean(BL_test(1:30, :, :),[2,3]));
ACC6 = squeeze(acur_test(:,1:30));
LOSS6 = squeeze(loss_test(:,1:30));

load(['./proposed/rand/Rand2_C18_E10_R50_B32.mat']);
BL5 = squeeze(BL_test(30, :, :));
B5 = squeeze(mean(BL_test(1:30, :, :),[2,3]));
ACC5 = squeeze(acur_test(:,1:30));
LOSS5 = squeeze(loss_test(:,1:30));

load(['./proposed/rand/Rand4_C9_E10_R50_B32.mat']);
BL4 = squeeze(BL_test(30, :, :));
B4 = squeeze(mean(BL_test(1:30, :, :),[2,3]));
ACC4 = squeeze(acur_test(:,1:30));
LOSS4 = squeeze(loss_test(:,1:30));

figure;
hold on;
box on;
grid on;
xlabel('Prediction instant $T$', 'interpreter', 'latex','fontsize',14);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',14);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL6, 1)), '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL5, 1)), 'm-^', 'linewidth', 1.5);
plot(0.1 : 0.1 : 0.9, squeeze(mean(BL4, 1)), 'b-square', 'linewidth', 1.5);

legend('$K=36$', '$K=18$',...
    '$K=9$',...
    'interpreter', 'latex',...
    'Location','southwest','fontsize',12);

figure;
hold on;
box on;
grid on;
xlabel('Comunication round', 'interpreter', 'latex','fontsize',16);
ylabel('Accuracy', 'interpreter', 'latex','fontsize',16);
plot(ACC6, '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(ACC5, 'm-^', 'linewidth', 1.5);
plot(ACC4, 'b-square', 'linewidth', 1.5);

legend('$K=36$', '$K=18$',...
    '$K=9$',...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',16);

figure;
hold on;
box on;
grid on;
xlabel('Comunication round', 'interpreter', 'latex','fontsize',16);
ylabel('Normalized beamforming gain $G_{\rm{N}}$', 'interpreter', 'latex','fontsize',16);
plot(B6, '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(B5, 'm-^', 'linewidth', 1.5);
plot(B4, 'b-square', 'linewidth', 1.5);

legend('$K=36$', '$K=18$',...
    '$K=9$',...
    'interpreter', 'latex',...
    'Location','southeast','fontsize',16);


figure;
hold on;
box on;
grid on;
xlabel('Comunication round', 'interpreter', 'latex','fontsize',16);
ylabel('Cross-Entropy Loss','fontsize',16);
plot(LOSS6, '-x','Color',[1    0.6   0.07], 'linewidth', 1.5);
plot(LOSS5, 'm-^', 'linewidth', 1.5);
plot(LOSS4, 'b-square', 'linewidth', 1.5);

legend('$K=36$', '$K=18$',...
    '$K=9$',...
    'interpreter', 'latex',...
    'Location','northeast','fontsize',16);




clear all;
close all;
clc;

% MM antenna num
MM_narrow_beam_antenna_num = 64;
% MM narrow beam num
MM_narrow_beam_num = 64;
% angular range
sector_start = - pi;
sector_end = pi;
% narrow beam generation
candidate_narrow_beam_angle = sector_start + (sector_end - sector_start) / MM_narrow_beam_num * [0.5 : 1 : MM_narrow_beam_num - 0.5];
candidate_narrow_beam = exp(-1i * [0 : MM_narrow_beam_antenna_num - 1]' * candidate_narrow_beam_angle) / sqrt(MM_narrow_beam_num);

% UE distribution
min_row = 1;
max_row = 999;
row_index = [min_row : max_row];
% load and save MM channel into MM_ch, row by row
MM_ch = zeros(length(row_index), 181, 64);
count = 1;
for i = row_index
    MM_file = ['DeepMIMO/MM_dataset_1/MM_DeepMIMO_dataset_' num2str(i) '_row.mat'];
    load(MM_file);
    % beam training results
    MM_ch(count, :, :) = squeeze(MM_channel(1, :, :)) * candidate_narrow_beam;
    count = count + 1;
end

train_size = 2;
valid_size = 6;
test_size = 2;
file_size = train_size+valid_size;
Q = 9;
m = 9;
K = 36;

samples = (Q+1)*(m+1)+1
during = (samples-1)*0.016
speeds = [5 : 5 : 30];
% UE acceleration
a_max = speeds * 0.2;

% MM beam training received signal
MM_data = zeros(file_size, 2, samples, MM_narrow_beam_num);
% MM optimal beam index
beam_label = zeros(file_size, samples);
% MM beam amplitude
beam_power = zeros(file_size, samples, MM_narrow_beam_num);

% for different UE speeds
for speed = speeds
    % make dictionary
    mkdir(['fixed_velocity_v' num2str(speed) '/train']);
    mkdir(['fixed_velocity_v' num2str(speed) '/valid']);
    mkdir(['fixed_velocity_v' num2str(speed) '/test']);
    for i = 1 : K
        min_l = mod(i,6)*100+100;
        max_l = mod(i,6)*100+400;
        for j = 1 : file_size
            % find UE trajectory within the pre-defined range
            flag = 0;
            while flag == 0
                initial_x = round(200 + rand * 600);
                initial_y = round(rand * 181);
                direction = rand * 2 * pi;
                a = rand * speed * 0.2;

                location = round([initial_x, initial_y] + (speed / 0.2 * [0 : 0.016 : during]' + ...
                    0.5 * a / 0.2 * ([0 : 0.016 : during] .^ 2)') * [cos(direction), sin(direction)]);
                if min(location(:, 1)) >= min_l && max(location(:, 1)) <= max_l && ...
                    min(location(:, 2)) >= 1 && max(location(:, 2)) <= 181
                    flag = 1;
                end
            end

            for k = 1 : samples
                [~, beam_label(j, k)] = max(squeeze(MM_ch(location(k, 1) - min_row+1,location(k, 2), :)));
                MM_data(j, 1, k, :) = real(MM_ch(location(k, 1) - min_row+1,location(k, 2), :));
                MM_data(j, 2, k, :) = imag(MM_ch(location(k, 1) - min_row+1,location(k, 2), :));
                beam_power(j, k, :) = abs(MM_ch(location(k, 1) - min_row+1,location(k, 2), :));
            end
        end
        MM_data_a = awgn(MM_data, 104);
        beam_power_a = beam_power;
        beam_label_a = beam_label;
        MM_data = MM_data_a(1:train_size,:,:,:);
        beam_label = beam_label_a(1:train_size,:);
        beam_power = beam_power_a(1:train_size,:,:);
        save(['fixed_velocity_v' num2str(speed) '/train/v' num2str(speed) '_Q' num2str(Q) '_k' num2str(i) '.mat'], ...
            'MM_data', 'beam_label', 'beam_power');

        MM_data = MM_data_a(train_size+1:valid_size,:,:,:);
        beam_label = beam_label_a(train_size+1:valid_size,:);
        beam_power = beam_power_a(train_size+1:valid_size,:,:);
        save(['fixed_velocity_v' num2str(speed) '/valid/v' num2str(speed) '_Q' num2str(Q) '_k' num2str(i) '.mat'], ...
            'MM_data', 'beam_label', 'beam_power');
    end
    min_l = 1;
    max_l = 999;

    MM_data = zeros(test_size, 2, samples, MM_narrow_beam_num);
    beam_label = zeros(test_size, samples); 
    beam_power = zeros(test_size, samples, MM_narrow_beam_num);

    for j = 1 : test_size
        flag = 0;
        while flag == 0
            initial_x = round(200 + rand * 600);
            initial_y = round(rand * 181);
            direction = rand * 2 * pi;
            a = rand * speed * 0.2;
            location = round([initial_x, initial_y] + (speed / 0.2 * [0 : 0.016 : during]' + ...
                    0.5 * a / 0.2 * ([0 : 0.016 : during] .^ 2)') * [cos(direction), sin(direction)]);
            if min(location(:, 1)) >= min_l && max(location(:, 1)) <= max_l && ...
                    min(location(:, 2)) >= 1 && max(location(:, 2)) <= 181
                    flag = 1;
            end
        end
        for k = 1 : samples
            [~, beam_label(j, k)] = max(squeeze(MM_ch(location(k, 1) - min_row+1,location(k, 2), :)));
            MM_data(j, 1, k, :) = real(MM_ch(location(k, 1) - min_row+1,location(k, 2), :));
            MM_data(j, 2, k, :) = imag(MM_ch(location(k, 1) - min_row+1,location(k, 2), :));
            beam_power(j, k, :) = abs(MM_ch(location(k, 1) - min_row+1,location(k, 2), :));
        end
    end
    MM_data = awgn(MM_data, 104);
    save(['fixed_velocity_v' num2str(speed) '/test/v' num2str(speed) '_Q' num2str(Q) '_test.mat'], ...
            'MM_data', 'beam_label', 'beam_power');
end

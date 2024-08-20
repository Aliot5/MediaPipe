clc; close all; clearvars;

file_x = str2double(readlines("./Data/data_x"));
file_y = str2double(readlines("./Data/data_y"));
file_r0 = str2double(readlines("./Data/data_r0"));

t = 1:334;

alpha = 0.3;
exponentialMA = filter(alpha, [1 alpha-1], file_x(:,1));
fig1 = figure;
movegui(fig1, "northwest")
plot(t, file_x(:,1), ...
     t, exponentialMA);
hold on;
title("smoothing angle_x"); xlabel("time"); ylabel("Angle");
legend("without smoothing", "with smoothing");
hold off;

alpha = 0.3;
exponentialMA = filter(alpha, [1 alpha-1], file_y(:,1));
fig2 = figure;
movegui(fig2, "north")
plot(t, file_y(:,1), ...
     t, exponentialMA);
hold on;
title("smoothing angle_y"); xlabel("time"); ylabel("Angle");
legend("without smoothing", "with smoothing");
hold off;

alpha = 0.3;
exponentialMA = filter(alpha, [1 alpha-1], file_r0(:,1));
fig3 = figure;
movegui(fig3, "northeast")
plot(t, file_r0(:,1), ...
     t, exponentialMA);
hold on;
title("smoothing angle_r_0"); xlabel("time"); ylabel("Angle");
legend("without smoothing", "with smoothing");
hold off;
clc;
close all;
clear all;

dir_to_search2 = 'C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\MCYT_resampled1\0006';
dir_to_search = 'C:\Users\andra\Documents\Egyetem\Allamvizsga\Adat\MCYT\0006';

csvpattern = fullfile(dir_to_search, '*.csv');
dinfo = dir(csvpattern);

csvpattern2 = fullfile(dir_to_search2, '*.csv');
dinfo2 = dir(csvpattern2);


for i = 1 : length(dinfo)
    fileName = fullfile(dir_to_search, dinfo(i).name);
    csv = csvread(fileName, 2, 0);
    X = csv(:, 1);
    Y = csv(:, 2);
    P = csv(:, 3);
    
    X1 = [];
    Y1 = [];
    for j = 2 : length(X)
        X1 = [X1, X(j) - X(j-1)];
        Y1 = [Y1, Y(j) - Y(j-1)];
    end
    
    fileName2 = fullfile(dir_to_search2, dinfo2(i).name);
    csv2 = csvread(fileName2, 2, 0);
    X2 = csv2(:, 1);
    Y2 = csv2(:, 2);
    P2 = csv2(:, 3);
       
    figure
    plot(X, Y, '-');
    title('Az aláírás')
    xlabel('X');
    ylabel('Y');
    
    figure
    subplot(2, 1, 1)
    plot(X1)
    title('X1 értékének váltakozása');
    xlabel('t');
    ylabel('X1');
    
    subplot(2, 1, 2)
    plot(Y1)
    title('Y1 értékének váltakozása');
    xlabel('t');
    ylabel('Y1');
    waitforbuttonpress;
end


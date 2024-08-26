clear
close all 
clc 

%%

desktopPath = fullfile(getenv('USERPROFILE'), 'Desktop');

load(fullfile(desktopPath, 'main_step_ICON', 'Workspace', 'step_ICON_test_results.mat'))
load(fullfile(desktopPath, 'main_step_ICON', 'Workspace', 'step_ICON_test_predicted_results.mat'))

load(fullfile(desktopPath, 'main_step_BEST_MATCH', 'Workspace', 'step_BEST_MATCH_test_predicted_results.mat'))
load(fullfile(desktopPath, 'main_step_GFS', 'Workspace', 'step_GFS_test_predicted_results.mat'))
load(fullfile(desktopPath, 'main_step_GEM', 'Workspace', 'step_GEM_test_predicted_results.mat'))
load(fullfile(desktopPath, 'main_step_METEO_FRANCE', 'Workspace', 'step_METEO_FRANCE_test_predicted_results.mat'))

load(fullfile(desktopPath, 'main_step_ICON', 'Workspace', 'step_ICON_test_r2_results.mat'))
load(fullfile(desktopPath, 'main_step_BEST_MATCH', 'Workspace', 'step_BEST_MATCH_test_r2_results.mat'))
load(fullfile(desktopPath, 'main_step_GFS', 'Workspace', 'step_GFS_test_r2_results.mat'))
load(fullfile(desktopPath, 'main_step_GEM', 'Workspace', 'step_GEM_test_r2_results.mat'))
load(fullfile(desktopPath, 'main_step_METEO_FRANCE', 'Workspace', 'step_METEO_FRANCE_test_r2_results.mat'))

%%

% Combine errors into a matrix for box plot
bestMatchR2AdjustedFirst24h = step_BEST_MATCH_test_r2_results(2,:);
bestMatchR2AdjustedSecond24h = step_BEST_MATCH_test_r2_results(4,:);
bestMatchR2Adjusted48h = step_BEST_MATCH_test_r2_results(6,:);

gsfGlobalR2AdjustedFirst24h = step_GFS_test_r2_results(2,:);
gsfGlobalR2AdjustedSecond24h = step_GFS_test_r2_results(4,:);
gsfGlobalR2Adjusted48h = step_GFS_test_r2_results(6,:);

gEMGlobalR2AdjustedFirst24h = step_GEM_test_r2_results(2,:);
gEMGlobalR2AdjustedSecond24h = step_GEM_test_r2_results(4,:);
gEMGlobalR2Adjusted48h = step_GEM_test_r2_results(6,:);

meteoFranceR2AdjustedFirst24h = step_METEO_FRANCE_test_r2_results(2,:);
meteoFranceR2AdjustedSecond24h = step_METEO_FRANCE_test_r2_results(4,:);
meteoFranceR2Adjusted48h = step_METEO_FRANCE_test_r2_results(6,:);

iconR2AdjustedFirst24h = step_ICON_test_r2_results(2,:);
iconR2AdjustedSecond24h = step_ICON_test_r2_results(4,:);
iconR2Adjusted48h = step_ICON_test_r2_results(6,:);

%%

figure
tiledlayout(2,2)

nexttile
% Creating the box plot
boxplot([bestMatchR2AdjustedSecond24h', ...
    iconR2AdjustedSecond24h', ...
    gsfGlobalR2AdjustedSecond24h', ...
    gEMGlobalR2AdjustedSecond24h', ...
    meteoFranceR2AdjustedSecond24h'],'Labels',{'Best match', 'ICON', 'GSF global', 'GEM global', 'Meteo France'})
ylabel('R2 adjusted');
title('Next day 24h forecast R2 adjusted');

nexttile
% Creating the box plot
boxplot([bestMatchR2AdjustedFirst24h', ...
    iconR2AdjustedFirst24h', ...
    gsfGlobalR2AdjustedFirst24h', ...
    gEMGlobalR2AdjustedFirst24h', ...
    meteoFranceR2AdjustedFirst24h'],'Labels',{'Best match', 'ICON', 'GSF global', 'GEM global', 'Meteo France'})
ylabel('R2 adjusted');
title('Fist day 24h forecast R2 adjusted');

nexttile

% Creating the box plot
boxplot([bestMatchR2Adjusted48h', ...
    iconR2Adjusted48h', ...
    gsfGlobalR2Adjusted48h', ...
    gEMGlobalR2Adjusted48h', ...
    meteoFranceR2Adjusted48h'],'Labels',{'Best match', 'ICON', 'GSF global', 'GEM global', 'Meteo France'})
ylabel('R2 adjusted');
title('48h forecast R2 adjusted');


%%

bestMatchR2First24h = step_BEST_MATCH_test_r2_results(1,:);
bestMatchR2Second24h = step_BEST_MATCH_test_r2_results(3,:);
bestMatchR248h = step_BEST_MATCH_test_r2_results(5,:);

gsfGlobalR2First24h = step_GFS_test_r2_results(1,:);
gsfGlobalR2Second24h = step_GFS_test_r2_results(3,:);
gsfGlobalR248h = step_GFS_test_r2_results(5,:);

gEMGlobalR2First24h = step_GEM_test_r2_results(1,:);
gEMGlobalR2Second24h = step_GEM_test_r2_results(3,:);
gEMGlobalR248h = step_GEM_test_r2_results(5,:);

meteoFranceR2First24h = step_METEO_FRANCE_test_r2_results(1,:);
meteoFranceR2Second24h = step_METEO_FRANCE_test_r2_results(3,:);
meteoFranceR248h = step_METEO_FRANCE_test_r2_results(5,:);

iconR2First24h = step_ICON_test_r2_results(1,:);
iconR2Second24h = step_ICON_test_r2_results(3,:);
iconR248h = step_ICON_test_r2_results(5,:);

%%


figure
% Creating the box plot
boxplot([bestMatchR2Second24h', ...
    iconR2Second24h', ...   
    gsfGlobalR2Second24h', ...
    gEMGlobalR2Second24h', ...
    meteoFranceR2Second24h'],'Labels',{'Best match', 'ICON', 'GSF global', 'GEM global', 'Meteo France'})
ylabel('R2');
title('Daily R^2');

fig = gcf;
fig.Units = 'inches';
fig.Position = [0 0 6 4];  % Adjust the size as needed
exportgraphics(gcf, 'daily_r2.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');


%%

figure
tiledlayout(2,2)

nexttile
% Creating the box plot
boxplot([bestMatchR2Second24h', ...
    iconR2Second24h', ...
    gsfGlobalR2Second24h', ...
    gEMGlobalR2Second24h', ...
    meteoFranceR2Second24h'],'Labels',{'Best match', 'ICON', 'GSF global', 'GEM global', 'Meteo France'})
ylabel('R2 ');
title('Next day 24h forecast R2 ');

nexttile
% Creating the box plot
boxplot([bestMatchR2First24h', ...
    iconR2First24h', ...
    gsfGlobalR2First24h', ...
    gEMGlobalR2First24h', ...
    meteoFranceR2First24h'],'Labels',{'Best match', 'ICON', 'GSF global', 'GEM global', 'Meteo France'})
ylabel('R2 ');
title('Fist day 24h forecast R2 ');

nexttile

% Creating the box plot
boxplot([bestMatchR248h', ...
    iconR248h', ...
    gsfGlobalR248h', ...
    gEMGlobalR248h', ...
    meteoFranceR248h'],'Labels',{'Best match', 'ICON', 'GSF global', 'GEM global', 'Meteo France'})
ylabel('R2 ');
title('48h forecast R2 ');


%%

s = 1;
y = 694;

actualPower = step_ICON_test_results(:, s:s+y);
predictedPowerBestMatch = step_BEST_MATCH_test_predicted_results(:, s:s+y);
predictedPowerGsfGlobal = step_GFS_test_predicted_results(:, s:s+y);
predictedPowerGEMGlobal = step_GEM_test_predicted_results(:, s:s+y);
predictedPowerMeteoFrance = step_METEO_FRANCE_test_predicted_results(:, s:s+y);

predictedPowerIcon = step_ICON_test_predicted_results(:, s:s+y);

%%

% Starting datetime
startDate = datetime(2023, 11, 1, 0, 0, 0);

% Hour data (from 0 to 200)
hours = 0:200;  % Adjust this range as needed

% Convert hours to datetime
dateTimes = startDate + hours/24;  % Dividing by 24 to convert hours to days

figure

plot(dateTimes, actualPower(1:201), 'b-');
hold on;
plot(dateTimes, predictedPowerBestMatch(1:201), 'r-');
hold on;
plot(dateTimes, predictedPowerIcon(1:201), 'y--');
hold on;
plot(dateTimes, predictedPowerGsfGlobal(1:201), 'c--');
hold on;
plot(dateTimes, predictedPowerGEMGlobal(1:201), 'm--');
hold on;
plot(dateTimes, predictedPowerMeteoFrance(1:201), 'g--');

legend('Actual', 'Best match', 'ICON', 'GSF global', 'GEM global','Meteo France');
xlabel('Hours');
ylabel('Power (MW)');

% Setting exactly 4 ticks on the x-axis
% We calculate ticks that are evenly spaced across the entire range
numTicks = 5;
interval = floor(length(hours) / (numTicks - 1)); % Interval between ticks
selectedTicks = 1:interval:length(hours); % Indices for selected ticks

% Apply calculated ticks
xticks(dateTimes(selectedTicks));

% Format the x-axis with dates
xtickformat('yyyy-MM-dd HH:mm');  % Modify format as desired

fig = gcf;
fig.Units = 'inches';
fig.Position = [0 0 10 5];  % Adjust the size as needed
exportgraphics(gcf, 'forecasts.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');



%%

windowSize = 5;  % Define window size
simpleSmooth1 = filter(ones(1,windowSize)/windowSize, 1, actualPower(1:200));
simpleSmooth2 = filter(ones(1,windowSize)/windowSize, 1, predictedPowerBestMatch(1:200));
simpleSmooth3 = filter(ones(1,windowSize)/windowSize, 1, predictedPowerIcon(1:200));

simpleSmooth4 = filter(ones(1,windowSize)/windowSize, 1, predictedPowerGsfGlobal(1:200));
simpleSmooth5 = filter(ones(1,windowSize)/windowSize, 1, predictedPowerGEMGlobal(1:200));
simpleSmooth6 = filter(ones(1,windowSize)/windowSize, 1, predictedPowerMeteoFrance(1:200));


figure

plot(simpleSmooth1);
hold on;
plot(simpleSmooth3, 'r');
hold on;
plot(simpleSmooth2, '--')
hold on;
plot(simpleSmooth4, '--');
hold on;
plot(simpleSmooth5, '--');
hold on;
plot(simpleSmooth6, '--');

legend('Actual', 'ICON','Best match', 'GSF global', 'GEM global','Meteo France');
xlabel('Hour');
ylabel('Forecast');

fig = gcf;
fig.Units = 'inches';
fig.Position = [0 0 12 6];  % Adjust the size as needed
exportgraphics(gcf, 'compare_models.pdf', 'ContentType', 'vector', 'BackgroundColor', 'none');




%%
figure
tiledlayout(2,1)

nexttile
plot(actualPower);
hold on;
plot(predictedPowerBestMatch);
hold on;
plot(predictedPowerIcon);
legend('Actual', 'Best match', 'ICON');
xlabel('Hour');
ylabel('Forecast');

nexttile
plot(actualPower-predictedPowerIcon);

%%

testas = [actualPower', predictedPowerBestMatch'];


%%

% Calculate Absolute Errors for each model
errorModel1 = abs(actualPower - predictedPowerBestMatch);
errorModel2 = abs(actualPower - predictedPowerIcon);
errorModel5 = abs(actualPower - predictedPowerGsfGlobal);
errorModel6 = abs(actualPower - predictedPowerGEMGlobal);
errorModel7 = abs(actualPower - predictedPowerMeteoFrance);


% Combine errors into a matrix for box plot
errorData = [errorModel1, errorModel2, errorModel5, errorModel6, errorModel7]; % Add other error arrays

% Creating the box plot
boxplot([errorModel1', errorModel2', errorModel5', errorModel6', errorModel7'],'Labels',{'Best match', 'ICON', 'GSF global', 'GEM global', 'Meteo France'})
ylabel('Absolute Error');
title('Box Plot of Forecast Errors for Different Models');

%%

% Assuming 'actual' and 'forecast' are arrays with your data
% and 'hours' is an array indicating the hour of the day for each data point
actual = actualPower;
forecast = predictedPowerIcon;

% Adjust the hours array to match the length of your data, which is now 1x695
hours = repmat(1:24, 1, 28);  % This covers 672 hours for 28 full days
hours = [hours, 1:23];  % Add hours to reach 695, omitting the 24th hour of the last day

% Repeat the hours array to cover both actual and forecasted data (now 1x1390)
hours = repmat(hours, 1, 2);  % Repeat for actual and forecasted data

% Concatenate actual and forecasted data for plotting
data = [actual, forecast];  % Concatenate horizontally to form a 1x1390 array

% Create a grouping variable for 'Actual' and 'Forecast'
dataType = [repmat({'Actual'}, 1, 695), repmat({'Forecast'}, 1, 695)];

% Use boxplot with grouping variables
boxplot(data, {dataType, hours}, 'factorgap', 5, 'factorseparator', 1, 'Colors', 'rb');

xlabel('Hour of Day');
ylabel('Power Value');
title('Comparison of Actual and Forecasted by ICON Power by Hour');
legend({'Actual', 'Forecast'});  % Adjust legend as needed

%%


% Assuming 'actual' and 'forecast' are arrays with your data
% and 'hours' is an array indicating the hour of the day for each data point
actual = actualPower;
forecast = predictedPowerIcon;
forecastBestMatch = predictedPowerBestMatch;

% Adjust the hours array to match the length of your data, which is now 1x695
hours = repmat(1:24, 1, 28);  % This covers 672 hours for 28 full days
hours = [hours, 1:23];  % Add hours to reach 695, omitting the 24th hour of the last day

% Repeat the hours array to cover both actual and forecasted data (now 1x1390)
hours = repmat(hours, 1, 2);  % Repeat for actual and forecasted data

% Concatenate actual and forecasted data for plotting
data = [actual, actual - forecast, actual - forecastBestMatch];  % Concatenate horizontally to form a 1x1390 array

% Create a grouping variable for 'Actual' and 'Forecast'
dataType = [repmat({'Actual'}, 1, 695), repmat({'Difference ICON'}, 1, 695), repmat({'Difference BEST MATCH'}, 1, 695)];

% Use boxplot with grouping variables
boxplot([(actual - forecast)', (actual - forecastBestMatch)']);

xlabel('Hour of Day');
ylabel('Power Value');
title('Difference of Actual and Forecasted by ICON Power by Hour');
legend({'Difference'});  % Adjust legend as needed



%%

% Plotting
figure;
subplot(3,2,1);
plot(actualPower, predictedPowerBestMatch, 'o');
xlabel('Actual Power');
ylabel('Predicted Power - BestMatch');
title('Actual vs Predicted - BestMatch');

subplot(3,2,2);
plot(actualPower, predictedPowerGsfGlobal, 'o');
xlabel('Actual Power');
ylabel('Predicted Power - GSF global');
title('Actual vs Predicted - GSF global');

subplot(3,2,3);
plot(actualPower, predictedPowerIcon, 'o');
xlabel('Actual Power');

ylabel('Predicted Power - ICON');
title('Actual vs Predicted - ICON');

subplot(3,2,4);
plot(actualPower, predictedPowerGEMGlobal, 'o');
xlabel('Actual Power');
ylabel('Predicted Power - GEM Global');
title('Actual vs Predicted - GEM Global');

subplot(3,2,5);
plot(actualPower, predictedPowerMeteoFrance, 'o');
xlabel('Actual Power');
ylabel('Predicted Power - Meteo France');
title('Actual vs Predicted - Meteo France');

%%

% Convert all data to double for consistency
actualPower = double(actualPower);
predictedPowerBestMatch = double(predictedPowerBestMatch);
predictedPowerIcon = double(predictedPowerIcon);
predictedPowerGEMGlobal = double(predictedPowerGEMGlobal);
predictedPowerMeteoFrance = double(predictedPowerMeteoFrance);
predictedPowerGsfGlobal = double(predictedPowerGsfGlobal);

% Calculations for BestMatch
mseBestMatch = immse(actualPower, predictedPowerBestMatch);
rmseBestMatch = sqrt(mseBestMatch);
maeBestMatch = mean(abs(actualPower - predictedPowerBestMatch));
mapeBestMatch = mean(abs((actualPower - predictedPowerBestMatch) ./ actualPower)) * 100;
rSquaredBestMatch = 1 - sum((predictedPowerBestMatch - actualPower).^2) / sum((actualPower - mean(actualPower)).^2);
n = y;
p = 5;
rSquaredBestMatch_adjusted = 1 - (1 - rSquaredBestMatch) * (n - 1) / (n - p - 1);

rSquaredBestHour =  1 - ((predictedPowerBestMatch - actualPower).^2) / ((actualPower - mean(actualPower)).^2);

% Calculate the mean of actualPower
meanActualPower = mean(actualPower);

% Calculations for ICON
mseIcon = immse(actualPower, predictedPowerIcon);
rmseIcon = sqrt(mseIcon);
maeIcon = mean(abs(actualPower - predictedPowerIcon));
mapeIcon = mean(abs((actualPower - predictedPowerIcon) ./ actualPower)) * 100;
rSquaredIcon = 1 - sum((predictedPowerIcon - actualPower).^2) / sum((actualPower - mean(actualPower)).^2);
n = y;
p = 5;
rSquaredIcon_adjusted = 1 - (1 - rSquaredIcon) * (n - 1) / (n - p - 1);

% Calculations for GEM Global
mseGEMGlobal = immse(actualPower, predictedPowerGEMGlobal);
rmseGEMGlobal = sqrt(mseGEMGlobal);
maeGEMGlobal = mean(abs(actualPower - predictedPowerGEMGlobal));
mapeGEMGlobal = mean(abs((actualPower - predictedPowerGEMGlobal) ./ actualPower)) * 100;
rSquaredGEMGlobal = 1 - sum((predictedPowerGEMGlobal - actualPower).^2) / sum((actualPower - mean(actualPower)).^2);
n = y;
p = 5;
rSquaredGEMGlobal_adjusted = 1 - (1 - rSquaredGEMGlobal) * (n - 1) / (n - p - 1);


% Calculations for Meteo France
mseMeteoFrance = immse(actualPower, predictedPowerMeteoFrance);
rmseMeteoFrance = sqrt(mseMeteoFrance);
maeMeteoFrance = mean(abs(actualPower - predictedPowerMeteoFrance));
mapeMeteoFrance = mean(abs((actualPower - predictedPowerMeteoFrance) ./ actualPower)) * 100;
rSquaredMeteoFrance = 1 - sum((predictedPowerMeteoFrance - actualPower).^2) / sum((actualPower - mean(actualPower)).^2);
n = y;
p = 5;
rSquaredMeteoFrance_adjusted = 1 - (1 - rSquaredMeteoFrance) * (n - 1) / (n - p - 1);

% Calculations for Gsf Global
mseGsfGlobal = immse(actualPower, predictedPowerGsfGlobal);
rmseGsfGlobal = sqrt(mseGsfGlobal);
maeGsfGlobal = mean(abs(actualPower - predictedPowerGsfGlobal));
mapeGsfGlobal = mean(abs((actualPower - predictedPowerGsfGlobal) ./ actualPower)) * 100;
rSquaredGsfGlobal = 1 - sum((predictedPowerGsfGlobal - actualPower).^2) / sum((actualPower - mean(actualPower)).^2);
n = y;
p = 5;
rSquaredGsfGlobal_adjusted = 1 - (1 - rSquaredGsfGlobal) * (n - 1) / (n - p - 1);

% Calculate NMAE for each model
nmaeBestMatch = maeBestMatch / meanActualPower;
nmaeIcon = maeIcon / meanActualPower;
nmaeIcon2 = maeIcon2 / meanActualPower;
nmaeIcon3 = maeIcon3 / meanActualPower;
nmaeGEMGlobal = maeGEMGlobal / meanActualPower;
nmaeMeteoFrance = maeMeteoFrance / meanActualPower;
nmaeGsfGlobal = maeGsfGlobal / meanActualPower;



% Creating the comparison table
modelNames = {'BestMatch', 'ICON', 'GEM Global', 'Meteo France', 'Gsf Global'};
MSEs = [mseBestMatch, mseIcon, mseGEMGlobal, mseMeteoFrance, mseGsfGlobal];
RMSEs = [rmseBestMatch, rmseIcon, rmseGEMGlobal, rmseMeteoFrance, rmseGsfGlobal];
MAEs = [maeBestMatch, maeIcon, maeGEMGlobal, maeMeteoFrance, maeGsfGlobal];
MAPEs = [mapeBestMatch, mapeIcon, mapeGEMGlobal, mapeMeteoFrance, mapeGsfGlobal];
R2s = [rSquaredBestMatch, rSquaredIcon, rSquaredGEMGlobal, rSquaredMeteoFrance, rSquaredGsfGlobal];
R2s_adjusted = [rSquaredBestMatch_adjusted, rSquaredIcon_adjusted, rSquaredGEMGlobal_adjusted, rSquaredMeteoFrance_adjusted, rSquaredGsfGlobal_adjusted];
NMAEs = [nmaeBestMatch, nmaeIcon, nmaeGEMGlobal, nmaeMeteoFrance, nmaeGsfGlobal];

comparisonTable = table(modelNames', MSEs', RMSEs', MAEs', MAPEs', R2s', R2s_adjusted', NMAEs', ...
    'VariableNames', {'Model', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'R2 adjusted' 'NMAE'});

% Display the updated table
disp(comparisonTable);

%%

actual = actualPower;

% 2. Squared Error
squaredErrorBestMatch = (predictedPowerBestMatch - actual).^2;
squaredErrorIcon = (predictedPowerIcon - actual).^2;
squaredErrorIconWithout10m = (predictedPowerIconWithout10m - actual).^2;
squaredErrorMeteoFrance = (predictedPowerMeteoFrance - actual).^2;

% Combine errors for box plotting (ignoring the first value for ASE due to diff)
errors = [squaredErrorBestMatch(2:end); squaredErrorIcon(2:end); squaredErrorIconWithout10m(2:end); squaredErrorMeteoFrance(2:end)]';

% Prepare grouping variable for box plot
errorTypes = [repmat({'Squared Error Best Match'}, 1, length(squaredErrorBestMatch)-1);  % One less due to diff in ASE
              repmat({'Squared Error Icon'}, 1, length(squaredErrorIcon)-1);
              repmat({'Squared Error Icon Without 10m'}, 1, length(squaredErrorIconWithout10m)-1);
              repmat({'Squared Error Meteo France'}, 1, length(squaredErrorMeteoFrance)-1);
              ];

% Box plot of error metrics
boxplot(errors, errorTypes(:));
ylabel('Error Value');
title('Box Plot of Various Forecasting Error Metrics');
set(gca, 'XTickLabelRotation', 45);  % Rotate labels for better readability


%%


% Creating the box plot
boxplot([rSquaredBestMatch_adjusted', ...
    rSquaredIcon_adjusted', ...
    rSquaredGsfGlobal_adjusted', ...
    rSquaredGEMGlobal_adjusted', ...
    rSquaredMeteoFrance_adjusted'],'Labels',{'Best match', 'ICON', 'GSF global', 'GEM global', 'Meteo France'})
ylabel('Absolute Error');
title('Box Plot of Forecast Errors for Different Models');


%%

% 1. Absolute Error
absError = abs(forecast - actual);

% 2. Squared Error
squaredError = (forecast - actual).^2;

% 3. Absolute Scaled Error (ASE)
naiveErrors = abs(diff(actual));  % Naive forecast error (difference between consecutive actual values)
meanNaiveError = mean(naiveErrors);
absoluteScaledErrors = abs(forecast(2:end) - actual(2:end)) / meanNaiveError;  % Ignoring the first value due to diff

% 4. Normalized Error
normalizedErrors = (forecast - actual) / std(actual);

% 5. Logarithmic Error
epsilon = 1e-6;  % Small constant to avoid log(0)
logarithmicErrors = log(forecast + epsilon) - log(actual + epsilon);

% Combine errors for box plotting (ignoring the first value for ASE due to diff)
errors = [absError(2:end); squaredError(2:end); absoluteScaledErrors; normalizedErrors(2:end); logarithmicErrors(2:end)]';

% Prepare grouping variable for box plot
errorTypes = [repmat({'Absolute Error'}, 1, length(absError)-1);  % One less due to diff in ASE
              repmat({'Squared Error'}, 1, length(squaredError)-1);
              repmat({'Absolute Scaled Error'}, 1, length(absoluteScaledErrors));
              repmat({'Normalized Error'}, 1, length(normalizedErrors)-1);
              repmat({'Logarithmic Error'}, 1, length(logarithmicErrors)-1)];

% Box plot of error metrics
boxplot(errors, errorTypes(:));
ylabel('Error Value');
title('Box Plot of Various Forecasting Error Metrics');
set(gca, 'XTickLabelRotation', 45);  % Rotate labels for better readability

%%

figure

plot(actualPower(1:y));
hold on;
plot(predictedPowerBestMatch(1:y));
hold on;
plot(predictedPowerGsfGlobal(1:y));
hold on;
plot(predictedPowerIcon(1:y));
hold on;
plot(predictedPowerGEMGlobal(1:y));
hold on;
plot(predictedPowerMeteoFrance(1:y));
legend('Actual', 'Best match', 'GSF Global', 'ICON', 'GEM global', 'Meteo France');
xlabel('Hour');
ylabel('Power forecast');


%%

figure
tiledlayout(2,2)

nexttile
plot(actualPower(1:168));
hold on;
plot(predictedPowerBestMatch(1:168));
hold on;
plot(predictedPowerGsfGlobal(1:168));
hold on;
plot(predictedPowerIcon(1:168));
hold on;
plot(predictedPowerGEMGlobal(1:168));
hold on;
plot(predictedPowerMeteoFrance(1:168));
legend('Actual', 'Best match', 'GSF Global', 'ICON', 'GEM global', 'Meteo France');
xlabel('Hour');
ylabel('Week 1 forecast');

nexttile
plot(actualPower(169:336));
hold on;
plot(predictedPowerBestMatch(169:336));
hold on;
plot(predictedPowerGsfGlobal(169:336));
hold on;
plot(predictedPowerIcon(169:336));
hold on;
plot(predictedPowerGEMGlobal(169:336));
hold on;
plot(predictedPowerMeteoFrance(169:336));
legend('Actual', 'Best match', 'GSF Global', 'ICON', 'GEM global', 'Meteo France');
xlabel('Hour');
ylabel('Week 2 forecast');

nexttile
plot(actualPower(337:504));
hold on;
plot(predictedPowerBestMatch(337:504));
hold on;
plot(predictedPowerGsfGlobal(337:504));
hold on;
plot(predictedPowerIcon(337:504));
hold on;
plot(predictedPowerGEMGlobal(337:504));
hold on;
plot(predictedPowerMeteoFrance(337:504));
legend('Actual', 'Best match', 'GSF Global', 'ICON', 'GEM global', 'Meteo France');
xlabel('Hour');
ylabel('Week 3 forecast');

nexttile
plot(actualPower(505:672));
hold on;
plot(predictedPowerBestMatch(505:672));
hold on;
plot(predictedPowerGsfGlobal(505:672));
hold on;
plot(predictedPowerIcon(505:672));
hold on;
plot(predictedPowerGEMGlobal(505:672));
hold on;
plot(predictedPowerMeteoFrance(505:672));
legend('Actual', 'Best match', 'GSF Global', 'ICON', 'GEM global', 'Meteo France');
xlabel('Hour');
ylabel('Week 4 forecast');

clear
close all 
clc 

%%
subdirectoryName = 'main_step_ICON';

%% LOAD dataset

load data/raw_data.mat

% filter if needed
% raw_data_turbine_meteo = raw_data(raw_data.timestamp >= '2023-10-01 00:00', :);
raw_data_turbine_meteo = raw_data;

raw_data_turbine_meteo.timestamp.TimeZone = 'Europe/Vilnius';

%%

dataTable = raw_data_turbine_meteo;

% Assume dataTable is your table with the 'Timestamp' and other columns.

% Create a logical array where true represents non-NaN values, and false represents NaNs
nonNanMatrix = ~isnan(table2array(dataTable(:, 2:end))); % Exclude the 'Timestamp' column

% Calculate a score by counting non-NaN values in each row
nonNanCount = sum(nonNanMatrix, 2);

% Add the non-NaN count as the first column of the table
dataTable = addvars(dataTable, nonNanCount, 'Before', 1);

% Sort the rows by 'Timestamp' and then by 'nonNanCount' in descending order
dataTable = sortrows(dataTable, {'timestamp', 'nonNanCount'}, {'ascend', 'descend'});

% Now use the 'unique' function to keep the first occurrence of each timestamp
[~, uniqueIdx] = unique(dataTable.timestamp, 'stable'); % 'stable' ensures the original order is preserved

% Select the unique rows base VV,,., :);

% Remove rows where 'nonNanCount' is zero
dataTable(dataTable.nonNanCount == 0, :) = [];

% Optionally, remove the 'nonNanCount' column if it's no longer needed
dataTable.nonNanCount = [];

clear nonNanMatrix nonNanCount uniqueIdx


%%

dataTable = sortrows(dataTable, 'timestamp');
dataTable = fillmissing(dataTable, 'previous');

%%

train_data = dataTable(dataTable.timestamp >= '2023-05-01 00:00' & dataTable.timestamp < '2023-10-01 00:00', :);
test_data = dataTable(dataTable.timestamp >= '2023-10-01 00:00' , :);


%%

figure;
plot(test_data.wind_speed_10m_best_matchms);
hold on;
plot(test_data.wind_speed_10m_ecmwf_ifs04ms);
hold on;
plot(test_data.wind_speed_10m_metno_nordicms);
hold on;
plot(test_data.wind_speed_10m_gfs_globalms);
hold on;
plot(test_data.wind_speed_10m_icon_eums);
hold on;
plot(test_data.wind_speed_10m_gem_globalms);
hold on;
plot(test_data.wind_speed_10m_meteofrance_arpege_europems);
hold off;
legend('wind speed 10m best matchms', 'wind speed 10m ecmwf ifs04ms', 'wind speed 10m metno nordicms', 'wind speed eums', 'wind speed gem globalms', 'wind speed meteofrance arpege europems');
xlabel('Sequence Index');
ylabel('Wind 10 m');


%%

% "pressure_msl_best_matchhPa", 
% "surface_pressure_best_matchhPa",
% "wind_speed_10m_best_matchms",
% "wind_speed_80m_best_matchms", ...
% "wind_speed_120m_best_matchms", 
% "wind_speed_180m_best_matchms", 
% "wind_gusts_10m_best_matchms", 
% 
% "pressure_msl_ecmwf_ifs04hPa",
% "surface_pressure_ecmwf_ifs04hPa", ...
% "wind_speed_10m_ecmwf_ifs04ms", 
% 
% "pressure_msl_metno_nordichPa", 
% "surface_pressure_metno_nordichPa", 
% "wind_speed_10m_metno_nordicms", 
% "wind_gusts_10m_metno_nordicms", ...

% "pressure_msl_gfs_globalhPa", 
% "surface_pressure_gfs_globalhPa", 
% "wind_speed_10m_gfs_globalms", 
% "wind_speed_80m_gfs_globalms", 
% "wind_speed_120m_gfs_globalms", ...
% "wind_gusts_10m_gfs_globalms", 

% "pressure_msl_icon_euhPa", 
% "surface_pressure_icon_euhPa", 
% "wind_speed_10m_icon_eums", 
% "wind_speed_80m_icon_eums", ...
% "wind_speed_120m_icon_eums", 
% "wind_speed_180m_icon_eums", 
% "wind_gusts_10m_icon_eums", 
% 
% "pressure_msl_gem_globalhPa", 
% "surface_pressure_gem_globalhPa", ...
% "wind_speed_10m_gem_globalms", 
% "wind_speed_80m_gem_globalms", 
% "wind_speed_120m_gem_globalms", 
% "wind_gusts_10m_gem_globalms", ...

% "pressure_msl_meteofrance_arpege_europehPa", 
% "surface_pressure_meteofrance_arpege_europehPa", 
% "wind_speed_10m_meteofrance_arpege_europems", ...
% "wind_speed_80m_meteofrance_arpege_europems", 
% "wind_speed_120m_meteofrance_arpege_europems", 
% "wind_speed_180m_meteofrance_arpege_europems", ...
% "wind_gusts_10m_meteofrance_arpege_europems"

features = ["power", ...
        "wind_speed_80m_icon_eums",...       //wind_speed_80m
        "wind_gusts_10m_icon_eums", ...      //wind_gusts_10m
        "wind_speed_10m_icon_eums", ...      //wind_speed_10m
        "pressure_msl_icon_euhPa"]; 

input_sequence_length = 3;

%% TRAIN

train_data = rmmissing(train_data(:, features));

train_data_table = timetable2table(train_data);

train_data_selected = train_data_table(:, features);

[y_train_array] = create_train_output_sequences(train_data_selected, input_sequence_length);

% Normalize the input features in the training dataset (excluding the power field)
train_data_mean = mean(train_data_selected{:, features});
train_data_std = std(train_data_selected{:, features});

train_data_table{:, features} = (train_data_table{:,features} - train_data_mean) ./ train_data_std;
    
train_data_selected = train_data_table(:, features);

[X_train_array] = create_train_input_sequences(train_data_selected, input_sequence_length);


%%

[X_test_sequences] = test_data_sequences_48h(test_data, features);

%%

search_space = [
    optimizableVariable('NumLayers', [1, 5], 'Type', 'integer');
    optimizableVariable('NumHiddenUnits', [10, 400], 'Type', 'integer');
    optimizableVariable('MaxEpochs', [10, 500], 'Type', 'integer');
    optimizableVariable('InitialLearnRate', [1e-4, 1e-1], 'Transform', 'log');
    optimizableVariable('MiniBatchSize', [8, 228], 'Type', 'integer');
    optimizableVariable('DropoutRate', [0, 0.6], 'Type', 'real');
    optimizableVariable('GradientThreshold', [1, 10], 'Type', 'real');
    optimizableVariable('L2Regularization', [1e-10, 1e-2], 'Transform', 'log');
    optimizableVariable('ClipGradients', {'true', 'false'}, 'Type', 'categorical');
    optimizableVariable('LearningRateSchedule', {'piecewise', 'none'}, 'Type', 'categorical');
    optimizableVariable('SequencePaddingValue', [0, 1], 'Type', 'real');
    optimizableVariable('SequenceLength', {'longest', 'shortest'}, 'Type', 'categorical');
];


%% Run Bayesian optimization

results = bayesopt(@(params) bilstm_objective(params, X_train_array, y_train_array, X_test_sequences, train_data_mean, train_data_std, input_sequence_length), ...
                   search_space, ...
                   'MaxObjectiveEvaluations', 30);

% Extract the best parameters
best_params = bestPoint(results);


%%

input_size = size(X_train_array{1}, 1);
output_size = size(y_train_array{1}, 1);

layers = [
    sequenceInputLayer(input_size)
];

for i = 1:best_params.NumLayers
    layers = [
        layers; 
        bilstmLayer(best_params.NumHiddenUnits, 'OutputMode', 'sequence');
        dropoutLayer(best_params.DropoutRate);
    ];
end

layers = [layers;
    fullyConnectedLayer(output_size)
    regressionLayer
];

if strcmp(best_params.ClipGradients, 'true')
    gradientThreshold = best_params.GradientThreshold; % Or whatever threshold you choose
else
    gradientThreshold = inf; % No clipping
end

sequenceLength = 'longest';
if strcmp(best_params.SequenceLength, 'longest')
    sequenceLength = 'longest';
elseif strcmp(best_params.SequenceLength, 'shortest')
    sequenceLength = 'shortest';
end

% Set training options
final_options = trainingOptions('adam', ...
    'ExecutionEnvironment','parallel',...
    'MaxEpochs', best_params.MaxEpochs, ...
    'InitialLearnRate', best_params.InitialLearnRate, ...
    'MiniBatchSize', best_params.MiniBatchSize, ...
    'GradientThreshold', gradientThreshold, ...
    'L2Regularization', best_params.L2Regularization, ...
    'Shuffle', 'every-epoch', ...
    'SequencePaddingValue', best_params.SequencePaddingValue, ...
    'SequenceLength', sequenceLength, ...
    'Verbose', 1);

% Train the final BiLSTM model with the best hyperparameters
final_model = trainNetwork(X_train_array, y_train_array, layers, final_options);


%% Evaluate the final model on the train set

y_train_predit = predict(final_model, X_train_array); 

y_train_matrix = cell2mat(y_train_array');
y_train_pred_matrix = cell2mat(y_train_predit');

% Calculate the performance metrics for the first value of each sequence
[r2, mse, rmse, mae, mape, bias] = fn_evaluate_model_performace('TEST model with TRAIN data', y_train_matrix, y_train_pred_matrix, length(features));


%% PLOT results

figure;
plot(y_train_matrix(:, 1000:2000));
hold on;
plot(y_train_pred_matrix(:, 1000:2000));
hold off;
legend('Train period power', 'Predicted power');
xlabel('Sequence Index');
ylabel('Power');


%%

y_test_matrix_all = [];
y_test_pred_matrix_all = [];
y_test_pred_matrix_all2 = [];

X_array = [];
X_array_pred = [];

for sn = 1:length(X_test_sequences)
    sequence = X_test_sequences{sn}
    test_data_table = (sequence - train_data_mean) ./ train_data_std;

    y_test_pred_array = [];
    y_test_pred_array2 = [];
    y_test_array = [];

    for i = 1:15

        start_idx = (i - 1) * input_sequence_length + 1;
        fprintf('start_idx: %.2f\n', start_idx);

        if(i < 4)

            % Extract the first X hours as history weather data
            history_data = test_data_table(start_idx:(start_idx + input_sequence_length - 1), 1:end);

            % Extract the next X hours as forecast weather data
            forecast_data = test_data_table(start_idx+input_sequence_length:(start_idx + 2*input_sequence_length - 1), 2:end);

            history_data_array = permute(table2array(history_data(:, :)), [2, 1]);  
            forecast_data_array = permute(table2array(forecast_data(:, :)), [2, 1]);

            X = [history_data_array; forecast_data_array];
            X2 = [history_data_array; forecast_data_array];
            X_array{sn}{i} = X;
        else

            % extract predicted data
            history_predicted_power = (cell2mat(y_test_pred_array(i-1)) - train_data_mean(1)) ./ train_data_std(1);

            % Extract the first X hours as history weather data
            history_data = test_data_table(start_idx:(start_idx + input_sequence_length - 1), 2:end);
            history_data2 = test_data_table(start_idx:(start_idx + input_sequence_length - 1), 1:end); %% just for testing

            % Extract the next X hours as forecast weather data
            forecast_data = test_data_table(start_idx+input_sequence_length:(start_idx + 2*input_sequence_length - 1), 2:end);

            history_data_array = permute(table2array(history_data(:, :)), [2, 1]);  
            forecast_data_array = permute(table2array(forecast_data(:, :)), [2, 1]);
            history_data_array2 = permute(table2array(history_data2(:, :)), [2, 1]);  

            X = [history_predicted_power; history_data_array; forecast_data_array];
            X2 = [history_data_array2; forecast_data_array];
            X_array{sn}{i} = X;
        end 

        y = permute(sequence(start_idx+input_sequence_length:(start_idx + 2*input_sequence_length - 1), 1).power, [2, 1]);

        y_test_predit = predict(final_model, X);
        y_test_predit2 = predict(final_model, X2);

        X_array_pred{sn}{i} = y_test_predit;

        y_test_pred_array{i} = y_test_predit;
        y_test_pred_array2{i} = y_test_predit2;

        y_test_array{i} = y;
    end

    y_test_matrix_all{sn} = y_test_array;
    y_test_pred_matrix_all{sn} = y_test_pred_array;
    y_test_pred_matrix_all2{sn} = y_test_pred_array2;

    TEMP_A = cell2mat(y_test_array);
    TEMP_A(1:3) = [];

    TEMP_B = cell2mat(y_test_pred_array);
    TEMP_B(1:3) = [];

    FIRST_24h_ACTUAL = TEMP_A(1:18);
    FIRST_24h_PREDICTED = TEMP_B(1:18);

    [r2, mse, rmse, mae, mape, bias, r2_adjusted] = fn_evaluate_model_performace('FIRST 24h', FIRST_24h_ACTUAL, FIRST_24h_PREDICTED, length(features));
   
    RESULTS(1,sn) = r2;
    RESULTS(2,sn) = r2_adjusted;

    SECOND_24h_ACTUAL = TEMP_A(19:end);
    SECOND_24h_PREDICTED = TEMP_B(19:end);

    [r2, mse, rmse, mae, mape, bias, r2_adjusted] = fn_evaluate_model_performace('SECOND 24h', SECOND_24h_ACTUAL, SECOND_24h_PREDICTED, length(features));

    RESULTS(3,sn) = r2;
    RESULTS(4,sn) = r2_adjusted;

    ALL_ACTUAL = TEMP_A;
    ALL_PREDICTED = TEMP_B;
    [r2, mse, rmse, mae, mape, bias, r2_adjusted] = fn_evaluate_model_performace('All 48 hours',TEMP_A, TEMP_B, length(features));

    RESULTS(5,sn) = r2;
    RESULTS(6,sn) = r2_adjusted;
end



%%

y_test_matrix_days = [];
y_test_pred_matrix_days = [];
y_test_pred_matrix_days2 = [];

for c = 1:length(y_test_matrix_all)
    y_test_matrix_days{c} = cell2mat(y_test_matrix_all{c}(8:end));
    y_test_pred_matrix_days{c} = cell2mat(y_test_pred_matrix_all{c}(8:end));
    y_test_pred_matrix_days2{c} = cell2mat(y_test_pred_matrix_all2{c}(8:end));
end


%% Calculate the performance metrics for the first value of each sequence

[r2, mse, rmse, mae, mape, bias] = fn_evaluate_model_performace('Calculate the performance metrics for the first value of each sequence', cell2mat(y_test_matrix_days), cell2mat(y_test_pred_matrix_days), length(features));

%%

test_results = cell2mat(y_test_matrix_days);
test_predicted_results = cell2mat(y_test_pred_matrix_days);
test_predicted_results2 = cell2mat(y_test_pred_matrix_days2);

figure
tiledlayout(1,1)

nexttile
plot(test_results(:, :));
hold on;
plot(test_predicted_results(:, :));
hold on;
plot(test_predicted_results2(:, :));
legend('Test period power', 'Predicted power with predicted power', 'Predicted power with actual power');
xlabel('Sequence Index');
ylabel('Power');

step_ICON_test_results = test_results;
step_ICON_test_predicted_results = test_predicted_results;
step_ICON_test_predicted_results2 = test_predicted_results2;
step_ICON_test_r2_results = RESULTS;

%% SAVE

% Save all variables
% fn_saveAllWorkspaceVars(subdirectoryName);

% Save all figures
% fn_saveAllFigures(subdirectoryName);


%% FUNCTIONS

function [X] = create_train_input_sequences(data, input_sequence_length)
    % Initialize the number of sequences based on the available data
    num_sequences = size(data, 1) - (2 * input_sequence_length) + 1;
    X = cell(num_sequences, 1);

    for i = 1:num_sequences
         % Extract the first X hours as history weather data
        history_data = data(i:(i + input_sequence_length - 1), 1:end);
        
        % Extract the next X hours as forecast weather data
        forecast_data = data(i+input_sequence_length:(i + 2*input_sequence_length - 1), 2:end);
        
        history_data_array = permute(table2array(history_data(:, :)), [2, 1]);  
        forecast_data_array = permute(table2array(forecast_data(:, :)), [2, 1]);

        % Combine history and forecast data into one array (24x20)
        X{i} = [history_data_array; forecast_data_array];
    end
end


function [y] = create_train_output_sequences(data, input_sequence_length)
    num_sequences = size(data, 1) - (2 * input_sequence_length) + 1;
    y = cell(num_sequences, 1); 
    for i = 1:num_sequences
        % Target values (energy feature for the next X hours)
        y{i} = permute(data(i+input_sequence_length:(i + 2*input_sequence_length - 1), 1).power, [2, 1]);
    end
end

function [R2, MSE, RMSE, MAE, MAPE, Bias, R2_adjusted] = fn_evaluate_model_performace(title, y_actual, y_predicted, feature_count)
    % Calculate R2 using the corrcoef function
    r = corrcoef(y_actual, y_predicted);
    R2 = r(1, 2) ^ 2;
    MSE = mean((y_actual - y_predicted).^2);
    RMSE = sqrt(MSE);
    MAE = mean(abs(y_actual - y_predicted));
    
    epsilon = 1e-1; % Small constant to avoid division by zero
    n = length(y_actual); % Number of observations
    
    % Calculation of MAPE with modification to handle zero or near-zero values
    MAPE = (100/n) * sum(abs(((y_actual) - (y_predicted)) ./ (y_actual + epsilon)));


    R2_adjusted = 1 - (1 - R2) * (n - 1) / (n - feature_count - 1);

    Bias = mean(y_predicted - y_actual);

    fprintf('\n\n');
    fprintf('Display the performance metrics\n');
    fprintf('TITLE: %s:\n', title);
    fprintf('R2: %.2f\n', R2);
    fprintf('R2 adjusted: %.2f\n', R2_adjusted);
    fprintf('MSE: %.2f\n', MSE);
    fprintf('RMSE: %.2f\n', RMSE);
    fprintf('MAE: %.2f\n', MAE);
    fprintf('MAPE: %.2f%%\n', MAPE);
    fprintf('Bias: %.2f\n', Bias);
    fprintf('Epsilon: %.2f\n', epsilon);
    fprintf('\n\n');
end

function sequences = test_data_sequences_48h(data, features)
    % number of hours for each sequence
    sequence_hours = 48;

    % total number of data points
    total_points = height(data);

    % Find rows where time is 00:00
    start_rows = find(hour(data.timestamp) == 0);

    % number of sequences that can be made
    num_sequences = floor(length(start_rows));

    % preallocate a cell array to hold the sequences
    sequences = cell(1, num_sequences);

    % loop through the data creating sequences
    for i = 1:num_sequences
        % calculate the indices for the start and end of the current sequence
        start_index = start_rows(i);
        end_index = start_index + sequence_hours - 1; % -1 as we include the start index
    
        % if end_index exceeds total points, break the loop
        if end_index > total_points
            break;
        end

        % Let's assume your timestamp column is named 'Timestamp' and the table is 'data'

        sequence = data(start_index:end_index, features);
        
        % First, sort the table by timestamp to ensure the data is in the right order
        sequence = sortrows(sequence, 'timestamp');

        % extract the current sequence and store it in the cell array
        sequences{i} = sequence;
    end

    % remove empty cells if any
    sequences = sequences(~cellfun('isempty',sequences));
end

% Create an objective function
function loss = bilstm_objective(params, X_train, y_train, X_test_sequences, train_data_mean, train_data_std, input_sequence_length)
    input_size = size(X_train{1}, 1);
    output_size = size(y_train{1}, 1);

    % Create the model
    layers = [
        sequenceInputLayer(input_size)
    ];
    
    for i = 1:params.NumLayers
        layers = [
            layers; 
            bilstmLayer(params.NumHiddenUnits, 'OutputMode', 'sequence');
            dropoutLayer(params.DropoutRate);
        ];
    end

    layers = [layers;
        fullyConnectedLayer(output_size)
        regressionLayer
    ];

    if strcmp(params.ClipGradients, 'true')
        gradientThreshold = params.GradientThreshold; % Or whatever threshold you choose
    else
        gradientThreshold = inf; % No clipping
    end

    sequenceLength = 'longest';
    if strcmp(params.SequenceLength, 'longest')
        sequenceLength = 'longest';
    elseif strcmp(params.SequenceLength, 'shortest')
        sequenceLength = 'shortest';
    end

    % Set training options
    options = trainingOptions('adam', ...
        'ExecutionEnvironment','parallel',...
        'MaxEpochs', params.MaxEpochs, ...
        'InitialLearnRate', params.InitialLearnRate, ...
        'MiniBatchSize', params.MiniBatchSize, ...
        'GradientThreshold', gradientThreshold, ...
        'L2Regularization', params.L2Regularization, ...
        'Shuffle', 'every-epoch', ...
        'SequencePaddingValue', params.SequencePaddingValue, ...
        'SequenceLength', sequenceLength, ...
        'Verbose', 0);

    % Train the BiLSTM model
    model = trainNetwork(X_train, y_train, layers, options);

    y_test_matrix_all = [];
    y_test_pred_matrix_all = [];
    
    for sn = 1:length(X_test_sequences)
        sequence = X_test_sequences{sn};
        test_data_table = (sequence - train_data_mean) ./ train_data_std;
    
        y_test_pred_array = [];
        y_test_array = [];
    
        for i = 1:15
    
            start_idx = (i - 1) * input_sequence_length + 1;
            % fprintf('start_idx: %.2f\n', start_idx);
    
            if(i < 4)
    
                % Extract the first X hours as history weather data
                history_data = test_data_table(start_idx:(start_idx + input_sequence_length - 1), 1:end);
        
                % Extract the next X hours as forecast weather data
                forecast_data = test_data_table(start_idx+input_sequence_length:(start_idx + 2*input_sequence_length - 1), 2:end);
        
                history_data_array = permute(table2array(history_data(:, :)), [2, 1]);  
                forecast_data_array = permute(table2array(forecast_data(:, :)), [2, 1]);
        
                X = [history_data_array; forecast_data_array];
            else
    
                % extract predicted data
                history_predicted_power = (cell2mat(y_test_pred_array(i-1)) - train_data_mean(1)) ./ train_data_std(1);
    
                % Extract the first X hours as history weather data
                history_data = test_data_table(start_idx:(start_idx + input_sequence_length - 1), 2:end);

                % Extract the next X hours as forecast weather data
                forecast_data = test_data_table(start_idx+input_sequence_length:(start_idx + 2*input_sequence_length - 1), 2:end);
        
                history_data_array = permute(table2array(history_data(:, :)), [2, 1]);  
                forecast_data_array = permute(table2array(forecast_data(:, :)), [2, 1]);

                X = [history_predicted_power; history_data_array; forecast_data_array];
            end 
    
            y = permute(sequence(start_idx+input_sequence_length:(start_idx + 2*input_sequence_length - 1), 1).power, [2, 1]);
    
            y_test_predit = predict(model, X);
    
            y_test_pred_array{i} = y_test_predit;
            y_test_array{i} = y;
        end
    
        y_test_matrix_all{sn} = y_test_array;
        y_test_pred_matrix_all{sn} = y_test_pred_array;
    end
    
    y_test_matrix_days = [];
    y_test_pred_matrix_days = [];
    
    for c = 1:length(y_test_matrix_all)
        y_test_matrix_days{c} = cell2mat(y_test_matrix_all{c}(8:end));
        y_test_pred_matrix_days{c} = cell2mat(y_test_pred_matrix_all{c}(8:end));
    end

    
    % Mean Squared Logarithmic Error (MSLE): This is useful when you want to penalize underestimates more than overestimates.
    error = meanSquaredLogarithmicError(cell2mat(y_test_matrix_days), cell2mat(y_test_pred_matrix_days));
    disp(['Mean Squared Logarithmic Error is: ', num2str(error)]);
    
    loss = error;

end

function msle = meanSquaredLogarithmicError(yTrue, yPred)
    % Ensure no negative values
    yTrue(yTrue < 0) = 0;
    yPred(yPred < 0) = 0;

    % Calculate the logarithmic differences
    logDiff = log(yTrue + 1) - log(yPred + 1);

    % Calculate the mean squared logarithmic error
    msle = mean(logDiff .^ 2);
end
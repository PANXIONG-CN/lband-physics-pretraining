function run_i2em_requests_matlab(request_csv, model_directory, output_csv)
% Run the official IEEE-GRSS I2EM reference implementation on a request CSV.
% The official I2EM_Backscatter_model.m is not redistributed by this project.

addpath(model_directory);
if exist('I2EM_Backscatter_model', 'file') ~= 2
    error('I2EM_Backscatter_model.m was not found in model_directory.');
end

requests = readtable(request_csv, 'TextType', 'string');
n = height(requests);
i2em_hh_db = nan(n, 1);
i2em_vv_db = nan(n, 1);

for index = 1:n
    if requests.correlation_model(index) == "exponential"
        spectrum_code = 1;
    elseif requests.correlation_model(index) == "gaussian"
        spectrum_code = 2;
    else
        error('Unsupported correlation model: %s', requests.correlation_model(index));
    end
    epsilon = complex(requests.dielectric_real(index), ...
        -requests.dielectric_loss_positive(index));
    [vv, hh, ~] = I2EM_Backscatter_model( ...
        requests.frequency_ghz(index), ...
        requests.rms_height_m(index), ...
        requests.correlation_length_m(index), ...
        requests.incidence_angle_deg(index), ...
        epsilon, spectrum_code, 1.5);
    i2em_hh_db(index) = hh;
    i2em_vv_db(index) = vv;
    fprintf('Completed I2EM request %d/%d\n', index, n);
end

results = table(requests.request_id, i2em_hh_db, i2em_vv_db, ...
    'VariableNames', {'request_id', 'i2em_hh_db', 'i2em_vv_db'});
writetable(results, output_csv);
end

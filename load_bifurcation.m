%%% loads the statistics (Pmin (mean of local minima), Pmax (mean of local maxima), Pmean etc.) 
function stats = load_bifurcation(base_folder, filename_pattern, resolution)

stats.Rmin = zeros(resolution,1);
stats.Rmax = zeros(resolution,1);
stats.Rmean = zeros(resolution,1);

stats.Cmin = zeros(resolution,1);
stats.Cmax = zeros(resolution,1);
stats.Cmean = zeros(resolution,1);

stats.Pmin = zeros(resolution,1);
stats.Pmax = zeros(resolution,1);
stats.Pmean = zeros(resolution,1);

for idx = resolution:-1:1

    filename = fullfile(base_folder, ...
        sprintf(filename_pattern, idx, resolution));

    s = load(filename);
    x = s.x(floor(end/2):end,:);
    % if no peaks, take max/min
    stats.Rmin(idx) = -mean(findpeaks(-x(floor(end/2):end,1))); % mean of local minima
    stats.Rmax(idx) = mean(findpeaks(x(floor(end/2):end,1))); % mean of local maxima
    stats.Rmean(idx) = mean(x(floor(end/2):end,1));

    stats.Cmin(idx) = -mean(findpeaks(-x(floor(end/2):end,2)));
    stats.Cmax(idx) = mean(findpeaks(x(floor(end/2):end,2)));
    stats.Cmean(idx) = mean(x(floor(end/2):end,2));

    if size(x,2) >= 3
        stats.Pmin(idx) = -mean(findpeaks(-x(floor(end/2):end,3)));
        stats.Pmax(idx) = mean(findpeaks(x(floor(end/2):end,3)));
        stats.Pmean(idx) = mean(x(floor(end/2):end,3));
    end

end
end

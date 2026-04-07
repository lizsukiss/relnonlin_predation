% plotting a coexistence matrix

%file = "C1lin_C2sat_coexistence_a1=1_a2=2.828427e+00_h2=8_resolution=30.mat"
%location = "./results/matrices/lin_sat/"
%baseDir = './results/matrices';

% Get all .mat files in all subfolders
%files = dir(fullfile(baseDir, '**', '*.mat'));

%[file,location] = uigetfile('*.mat');

%% r - c

[files, location] = uigetfile('*.mat', 'Select matrices', 'MultiSelect', 'on');
if isequal(files,0); return; end
if ischar(files); files = {files}; end   % single → cell

n = numel(files);

% sort by h2
h2vals = nan(n,1);

expr = ['_h2=(?<h2>[^_]+)'];

for k = 1:n
    tok = regexp(files{k}, expr, 'names','once');
    h2vals(k) = str2double(tok.h2);
end

[~, ord] = sort(h2vals);
files = files(ord);


ncol = ceil(sqrt(n));
nrow = ceil(n/ncol);

figure

for ii = 1:n
    
    fullpath = fullfile(location,files{ii})
    S = load(fullpath);        % load .mat file
    M = S.coexistence;      % loading the matrix
    

    subplot(nrow, ncol, ii)


    imagesc(M')
    axis image
    set(gca,'YDir','normal')
    
    % discrete colormap for 0/1/2
    cmap = [ ...
        1 1 1;    % 0 → white
        0.3 0.7 0.3;  % 1 → green
        0.8 0.2 0.2   % 2 → red
    ];
    
    colormap(cmap)
    caxis([0 2])
    
    % make NaNs gray
    set(gca,'Color',[0.7 0.7 0.7])   % background color
    set(findobj(gca,'Type','Image'),'AlphaData',~isnan(M'))
    
    
    % title
    [~, file] = fileparts(fullpath);
    
    expr = ...
    ['C1lin_C2(?<C2>lin|sat)', ...
     '(?:_P(?<P>lin|sat))?', ...        % ← optional P block
     '.*?_a2=(?<a2>[^_]+)_h2=(?<h2>[^_]+)', ...
     '(?:_aP=(?<aP>[^_]+))?'];          % aP optional too
    
    tok = regexp(file, expr, 'names','once');
    C2 = convertCharsToStrings(tok.C2)
    a2 = convertCharsToStrings(tok.a2)
    h2 = convertCharsToStrings(tok.h2)
    
    title(sprintf( ...
        'C1 linear – C2 %s | a_2 = %.3f, h_2 = %.3f', ...
        C2, a2, h2))

end

% one shared colorbar
cb = colorbar('Position',[0.92 0.15 0.02 0.7]);
cb.Ticks = [1/3 1 5/3];
cb.Limits = [0 2];
cb.TickLabels = {
    'no coexistence'
    'static equilibrium'
    'dynamic equilibrium'
};
cb.Label.Rotation = 90;


%% r - c1 - c2 - p

[files, location] = uigetfile('*.mat', 'Select matrices', 'MultiSelect', 'on');
if isequal(files,0); return; end
if ischar(files); files = {files}; end   % single → cell

n = numel(files);

% sort by some parameter
paramvals = nan(n,1);

expr = ['_a2=(?<param>[^_]+)']; % change the regex for another parameter 

for k = 1:n
    tok = regexp(files{k}, expr, 'names','once');
    paramvals(k) = str2double(tok.param);
end

[~, ord] = sort(paramvals);
files = files(ord);


ncol = ceil(sqrt(n));
nrow = ceil(n/ncol);

figure

for ii = 1:n
    
    fullpath = fullfile(location,files{ii})
    S = load(fullpath);        % load .mat file
    M = S.coexistence;      % loading the matrix
    

    subplot(nrow, ncol, ii)


    imagesc(M')
    axis image
    set(gca,'YDir','normal')
    
    % discrete colormap for 0/1/2
    cmap = [ ...
        1 1 1;    % 0 → white
        0.3 0.7 0.3;  % 1 → green
        0.8 0.2 0.2   % 2 → red
    ];
    
    colormap(cmap)
    caxis([0 2])
    
    % make NaNs gray
    set(gca,'Color',[0.7 0.7 0.7])   % background color
    set(findobj(gca,'Type','Image'),'AlphaData',~isnan(M'))
    
    
    % title
    [~, file] = fileparts(fullpath);
    
    expr = ...
    [  'C1lin_C2(?<C2>lin|sat)', ...
  '_P(?<P>lin|sat)', ...                      % ← mandatory P
  '.*?_a2=(?<a2>[^_]+)_h2=(?<h2>[^_]+)', ...
  '(?:_aP=(?<aP>[^_]+))?'];          % aP optional too
    
    tok = regexp(file, expr, 'names','once');
    C2 = convertCharsToStrings(tok.C2)
    P  = convertCharsToStrings(tok.P)
    a2 = convertCharsToStrings(tok.a2)
    h2 = convertCharsToStrings(tok.h2)
    
    title(sprintf( ...
        'C1 lin, C2 %s, P %s | a_2 = %.3f, h_2 = %.3f', ...
        C2, P, a2, h2))

end

% one shared colorbar
cb = colorbar('Position',[0.92 0.15 0.02 0.7]);
cb.Ticks = [1/3 1 5/3];
cb.Limits = [0 2];
cb.TickLabels = {
    'no coexistence'
    'static equilibrium'
    'dynamic equilibrium'
};
cb.Label.Rotation = 90;
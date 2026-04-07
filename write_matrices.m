% creating matrices based on individual runs

baseTarDir   = '.\results\lin_lin_P\';
baseMatDir   = '.\results\matrices\';
baseTmpDir   = '.\results\temp\';

a2_values = 2.^linspace(-2,5,15);
h2_values = 2.^linspace(-2,5,15);
aP_values = [1,8];

model_names = {'matrix_lin_lin_lin', 'matrix_lin_lin_sat', 'matrix_lin_sat_lin', 'matrix_lin_sat_sat', 'matrix_lin_sat'}

for ii = 1:length(model_names)

    model_name = model_names{ii}

    
    for a2 = a2_values
        for h2 = h2_values
            for aP = aP_values
               
               
                % look for the matrix file
                files = dir(fullfile(baseMatDir, '*.mat'));
                
    
                targetmatrix.a2 = a2;
                targetmatrix.h2 = h2;
                targetmatrix.aP = aP;
                
                matrixfile = find_best_match(files, targetmatrix, model_name);
                matrixfile = [baseMatDir, matrixfile];
    
                if ~isfile(matrixfile) % if exists, do nothing
                    
                    % look for the tar.gz file
                    tarfiles = dir(fullfile(baseTarDir, '*.tar.gz'));
                    targettar = find_best_match(tarfiles, targetmatrix, 'tar');
    
                    targettar = [baseTarDir,targettar];
    
                    if isfile(targettar)
    
                        % tar exists, untar and read in
                        if ~isfolder(baseTmpDir)
                            mkdir(baseTmpDir)
                        end
            
                        % untar
                        fprintf('Processing %s ...\n', targettar);
                    
                        try
                            untar(targettar, baseTmpDir);
                        catch ME
                            fprintf(2, 'Skipping corrupt tar: %s\n', targettar);
                            fprintf(2, '    Reason: %s\n', ME.message);
                            continue
                        end
                        %%%%%%
    
                        resolution = 30;
                        coexistence_matrix = nan(resolution, resolution); % resolution is 30
                        
                        d1_vector = linspace(0,1,32);
                        d1_vector = d1_vector(2:31);
                        d2_vector = linspace(0,a2/(1+a2*h2),32);
                        d2_vector = d2_vector(2:31);
    
                        for i = 1:resolution % d1 index
                            for j = 1:resolution % d2 index
                                
                                d1 = d1_vector(i);
                                d2 = d2_vector(j);
    
                                timeseriesfiles = dir([baseTmpDir,targettar(21:end-6),'/*.mat']);
                                
                                targettimeseries.d1 = d1;
                                targettimeseries.d2 = d2;
    
                                % look for the closest matching file
                                timeseriesFile = find_best_match(timeseriesfiles, targettimeseries, 'run');
                                timeseriesFile = [baseTmpDir,targettar(21:end-6),'\',timeseriesFile];
                        
                                if isempty(timeseriesFile)
                                    coexistence_matrix(i,j) = nan;
                                    "best match not found"
                                else
                                    S = load(timeseriesFile);   % contains x
                            
                                    x = S.x;
                            
                                    if any(isnan(x(:))) || any(x(:) < 0)
                                        coexistence_matrix(i,j) = nan;
                                        continue
                                    end
                            
                                    if all(x > 1e-10)
                                        cv = std(x)./mean(x);
                                        if all(cv < 0.01)
                                            coexistence_matrix(i,j) = 1;
                                        else
                                            coexistence_matrix(i,j) = 2;
                                        end
                                    end
                                end
                            end
                        end
                
                        coexistence = coexistence_matrix;
                        matrix_struct_to_be_saved = struct("coexistence", coexistence);
                        
                        if model_name == 'matrix_lin_lin_lin'
                            filename_matrix = sprintf('%sC1lin_C2lin_Plin_a1=1_a2=%s_h2=%s_aP=%s.mat', baseMatDir,num2str(a2),num2str(h2),num2str(aP));
                        elseif model_name == 'matrix_lin_lin_sat'
                            filename_matrix = sprintf('%sC1lin_C2lin_Psat_a1=1_a2=%s_h2=%s_aP=%s.mat', baseMatDir,num2str(a2),num2str(h2),num2str(aP));
                        elseif model_name == 'matrix_lin_sat_lin'
                            filename_matrix = sprintf('%sC1lin_C2sat_Plin_a1=1_a2=%s_h2=%s_aP=%s.mat', baseMatDir,num2str(a2),num2str(h2),num2str(aP)); 
                        elseif model_name == 'matrix_lin_sat_sat'
                            filename_matrix = sprintf('%sC1lin_C2sat_Psat_a1=1_a2=%s_h2=%s_aP=%s.mat', baseMatDir,num2str(a2),num2str(h2),num2str(aP)); 
                        elseif model_name == 'matrix_lin_sat'
                            filename_matrix = sprintf('%sC1lin_C2sat_a1=1_a2=%s_h2=%s.mat', baseMatDir,num2str(a2),num2str(h2)); 
                        end

                        save(filename_matrix, '-fromstruct', matrix_struct_to_be_saved)
                        
                        % clear temporary extraction
                        rmdir(baseTmpDir, 's')
                    end
                else
                    matrixfile
                end
            end
        end
    end
end % for model names    


function fname = find_best_match(files, target, mode)

    tol = 1e-3;
    bestScore = inf;
    fname = '';

    for k = 1:numel(files)
        p = parse_by_mode(files(k).name, mode);
        if isempty(p); continue; end

        switch mode
            case 'matrix'   % compare a2, h2, aP
                score = abs(p.a2 - target.a2) + ...
                        abs(p.h2 - target.h2) + ...
                        abs(p.aP - target.aP);

            case 'tar'      % compare a2, h2, aP
                score = abs(p.a2 - target.a2) + ...
                        abs(p.h2 - target.h2) + ...
                        abs(p.aP - target.aP);

            case 'run'      % compare a2, d1, d2
                score = abs(p.d1 - target.d1) + ...
                        abs(p.d2 - target.d2);
        end

        if score < bestScore && score < tol*10
            bestScore = score;
            fname = files(k).name;
        end
    end
end

function p = parse_by_mode(fname, mode)

    p = [];

    switch mode
        %% MATRICES
        case 'matrix_lin_lin_lin'
            expr = ['<C1lin_C2lin_Plin_coexistence_a1=1_a2=(?<a2>[^_]+)_h2=(?<h2>[^_]+)_', ...
                    'aP=(?<aP>[^_]+)_hP=(?<hP>[^_]+)_dP=(?<dP>[^_]+)'];
        
        case 'matrix_lin_lin_sat'
            expr = ['<C1lin_C2lin_Psat_coexistence_a1=1_a2=(?<a2>[^_]+)_h2=(?<h2>[^_]+)_', ...
                    'aP=(?<aP>[^_]+)_hP=(?<hP>[^_]+)_dP=(?<dP>[^_]+)'];
        %% TAR FILES
        case 'tar'
            expr = ['a2=(?<a2>[^_]+)_h2=(?<h2>[^_]+)_aP=(?<aP>[^\.]+)'];

        %% INDIVIDUAL RUNS
        case 'run'
            expr = ['a2=(?<a2>[^_]+)_aP=(?<aP>[^_]+)_', ...
                    'h1=[^_]+_h2=[^_]+_hP=[^_]+_', ...
                    'd1=(?<d1>[^_]+)_d2=(?<d2>[^_]+)'];
    end

    tok = regexp(fname, expr, 'names');
    if isempty(tok); return; end

    f = fieldnames(tok);
    for k = 1:numel(f)
        p.(f{k}) = str2double(tok.(f{k}));
    end
end

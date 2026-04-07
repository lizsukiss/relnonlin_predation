% function for making a coex plane for the basic (R-C1+C2) model

function coex_matrix = translate2matrix_basic(a1, a2, h2, resolution)
    
    filename_matrix = sprintf('./matrices/model_linsat/a2=%s_h2=%s_resolution=%s.mat',...
            num2str(a2), num2str(h2),num2str(resolution));
   
    if exist(filename_matrix,'file')
        coex_matrix = load(filename_matrix,'coexistence');
        coex_matrix = coex_matrix.coexistence;
    else
    
        d1_values = get_grid(a1,0,resolution);
        d2_values = get_grid(a2,h2,resolution);
        
        % C1 timeseries --> C2 invasion
        C2_invasion = zeros(resolution);
        base_folder = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f/normal', ...
            a1, 0, 0, 0, 0.1); % parameter dP hard-coded, should make no difference
    
        for d1_idx = 1:length(d1_values)
            
            filename = fullfile(base_folder, ...
                sprintf('d1_idx_%d_of_%d.mat', d1_idx, resolution));
            %disp('until here all good')
            RC1_data = load(filename);
            %disp('data loaded')
           
            % only regarding the second half of the simulations
            % if there was an error and it couldn't integrate for at least
            % 1000 steps, let's say it's nan
            if length(RC1_data.x) < 1000
                R = nan;
            else
                R = RC1_data.x(floor(end/2):end,1);
            end
            C2_invasion(d1_idx,:) = mean(a2*R./(1+a2*h2*R)) - d2_values; % timeseries not needed but since it is easier to simulate it, I'll keep it

        end
        
        C2_invasion(C2_invasion>0) = 2;
        C2_invasion(C2_invasion<0) = 0;
    
        % C2 timeseries --> C1 invasion
        C1_invasion = zeros(resolution);
       
        for d2_idx = 1:length(d2_values)
            base_folder = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f/reversed', ... % normal and reversed ought to be the same when P is not present but reversed could be more 'stable'
                                    a2, h2, 0, 0, 0.1); % parameter dP is hard-coded, should make no difference
          
            filename = fullfile(base_folder, ...
                sprintf('d2_idx_%d_of_%d.mat', d2_idx, resolution));
    
            RC2_data = load(filename);
            
            % only regarding the second half of the simulations
            % if there was an error and it couldn't integrate for at least
            % 1000 steps, let's say it's nan
            if length(RC2_data.x) < 1000
                R = nan;
            else
                R = RC2_data.x(floor(end/2):end,1);
            end
            
            C1_invasion(:,d2_idx) = mean(a1 * R) - d1_values;           
        
        end
    
        C1_invasion(C1_invasion > 0) = 1;
        C1_invasion(C1_invasion < 0) = 0;
    
        coex_matrix = C1_invasion + C2_invasion;
        
        % save
        matrix_struct_to_be_saved = struct("coexistence", coex_matrix);
        save(filename_matrix, '-fromstruct', matrix_struct_to_be_saved)
    end
end

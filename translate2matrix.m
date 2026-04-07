%%%% Transform subsystem simulations to coexistence matrices via invasion
%%%% growth rates

% Input: parameters and type of model (C1 is assumed to be the same for
% now, even if it is accepted as an input!)
% Simulation outcomes are read from saved data
% Check data for unfinished simulations (errors with tolerances etc.)
% Invasion growth rate is determined

function filename = translate2matrix(C1_parameters, C2_parameters, P_parameters, resolution, C2_model, direction) % C2_model is 'linearized' or 'rnl', direction is 'normal or 'reversed'

    if strcmp(direction,'normal')
        d1_direction = 'normal';
        if strcmp(C2_model,'linearized')
            d2_direction = 'linearized';
        else
            d2_direction = 'normal';
        end
    else
        d1_direction = 'reversed';
        if strcmp(C2_model,'linearized')
            d2_direction = 'linearized_reversed';
        else
            d2_direction = 'reversed';
        end
    end

    %setting all parameters
    a1 = C1_parameters.a1;
    h1 = C1_parameters.h1;
    a2 = C2_parameters.a2;
    h2 = C2_parameters.h2;
    aP = P_parameters.aP;
    hP = P_parameters.hP;
    dP = P_parameters.dP;

    % define filename and chechk if it already exists (if it does, skip computation)
    filename = sprintf('./matrices/a1=%.2f_a2=%.2f_h2=%.2f_aP=%.2f_hP=%.2f_dP=%.2f_C2=%s_dir=%s.mat',...
        a1,a2,h2,aP,hP,dP,C2_model,direction);

    if exist(filename,"file")
        fprintf('File %s already exists, skipping matrix definitions.',filename)
    else
            
        d1_values = get_grid(a1,h1,resolution);
        d2_values = get_grid(a2,h2,resolution);
       
        % C2 invasion --> based on R-C1-P time series
        C2_invasion = nan(resolution);
        base_folder = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
            a1, h1, aP, hP, dP);
        
        if strcmp(C2_model,'linearized')
            a2_values = (1 - d2_values*h2) * a2;
        end
    
        for d1_idx = 1:length(d1_values)
            
            filename_C1 = fullfile(base_folder, ...
                sprintf('%s/d1_idx_%d_of_%d.mat', d1_direction, d1_idx, resolution));
            
            RC1P_data = load(filename_C1);
    
            R = RC1P_data.x(:,1);
            C1 = RC1P_data.x(:,2);
            P = RC1P_data.x(:,3);
            
            if strcmp(C2_model,'linearized')
                for d2_idx = 1:length(d2_values)
                    C2_invasion(d1_idx,d2_idx) = mean(a2_values(d2_idx)*R - aP*P./(1+aP*hP*C1)) - d2_values(d2_idx);
                end
            else
                C2_invasion(d1_idx,:) = mean(a2*R./(1+a2*h2*R) - aP*P./(1+aP*hP*C1)) - d2_values;
            end
            
        end
        
        C2_invasion(C2_invasion>0) = 2;
        C2_invasion(C2_invasion<0) = 0;
    
        % C1 invasion --> R-C2-P time series
        C1_invasion = nan(resolution);
       
        for d2_idx = 1:length(d2_values)
            base_folder_1 = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
                                    a2, h2, aP, hP, dP);
              
            filename1 = fullfile(base_folder_1, ...
                    sprintf('%s/d2_idx_%d_of_%d.mat', d2_direction, d2_idx, resolution));
               
            RC2P_data_1 = load(filename1);
            
            R = RC2P_data_1.x(:,1);
            C2 = RC2P_data_1.x(:,2);
            P = RC2P_data_1.x(:,3);
                
            C1_invasion(:,d2_idx) = mean(a1 * R - aP*P./(1+aP*hP*C2)) - d1_values;           
            
        end
        C1_invasion(C1_invasion > 0) = 1;
        C1_invasion(C1_invasion < 0) = 0;
    
        % P invasion --> in R-C1, in R-C2 and in R-C1+C2
        P_invasion_in_C1 = nan(resolution);
        P_invasion_in_C2 = nan(resolution);
        P_invasion_in_C1C2 = nan(resolution);

        if aP == 0
            P_invasion_in_C1 = zeros(resolution);
            P_invasion_in_C2 = zeros(resolution);
            P_invasion_in_C1C2 = zeros(resolution);
            P_invasion_in_main = zeros(resolution);
        else           
        
            % in R-C2
            for d2_idx = 1:length(d2_values)
        
                base_folder = sprintf('./RC2P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
                                        a2, h2, 0, 0, dP); % compared to the system without the predator
                  
                filename_C2 = fullfile(base_folder, ...
                        sprintf('%s/d2_idx_%d_of_%d.mat', d2_direction, d2_idx, resolution));
                   
                RC2_data = load(filename_C2);
                
                C2 = RC2_data.x(:,2);
                    
                P_invasion_in_C2(:,d2_idx) = mean(aP * C2./(1+aP*hP*C2)) - dP + 0*d1_values; % +0 for size matching            
                
            end
        
            % in R-C1
            for d1_idx = 1:length(d2_values)
                
                base_folder = sprintf('./RC1P/a=%.2f_h=%.2f_aP=%.2f_hP=%.2f_dP=%.2f', ...
                a1, h1, 0, 0, dP);   % compared to the system without the predator
                  
                filename_C1 = fullfile(base_folder, ...
                        sprintf('%s/d1_idx_%d_of_%d.mat', d1_direction, d1_idx, resolution));
                   
                RC1_data = load(filename_C1);
                
                C1 = RC1_data.x(:,2);
                    
                P_invasion_in_C1(d1_idx,:) = mean(aP * C1./(1+aP*hP*C1)) - dP + 0*d2_values; % +0 for size matching            
                
            end
            
            % in R-C1C2
            C1C2_coex_matrix = translate2matrix_basic(a1,a2,h2,resolution);
            for d1_idx = 1:resolution
                for d2_idx = 1:resolution
                    % when coexistence is possible, open the relevant file and
                    % check the density of C1 and C2
                    
                    if C1C2_coex_matrix(d1_idx,d2_idx) == 3 % 1 if C1 can invade + 2 if C2 can
                        
                        base_folder = sprintf('./RC1C2/a1=%.2f_a2=%.2f_h2=%.2f', ...
                            a1, a2, h2);
        
                        filename_C1C2 = fullfile(base_folder, ...
                            sprintf('d1_idx_%d_of_%d_d2_idx_%d_of_%d.mat', d1_idx, resolution, d2_idx, resolution));
                        
                        density = load(filename_C1C2);
                        density = density.x;
                        
                        C1 = density(:,2);
                        C2 = density(:,3);
                        % invasion rates
                        P_invasion_in_C1C2(d1_idx,d2_idx) = mean(aP*(C1+C2)./(1+aP*hP*(C1+C2)))-dP;
                   
                    end
                end
            end
            
            P_invasion_in_C1(P_invasion_in_C1 > 0) = .5;
            P_invasion_in_C1(P_invasion_in_C1 < 0) = 0;
            
            P_invasion_in_C2(P_invasion_in_C2 > 0) = .6;
            P_invasion_in_C2(P_invasion_in_C2 < 0) = 0;
            
            P_invasion_in_C1C2(P_invasion_in_C1C2 > 0) = .7;
            P_invasion_in_C1C2(P_invasion_in_C1C2 < 0) = 0;
        
            P_invasion_in_main = nan(resolution);
        
            for d1_idx = 1:resolution
                for d2_idx = 1:resolution
                    if C1C2_coex_matrix(d1_idx,d2_idx) == 1
                        P_invasion_in_main(d1_idx,d2_idx) = P_invasion_in_C1(d1_idx,d2_idx);
                    elseif C1C2_coex_matrix(d1_idx,d2_idx) == 2
                        P_invasion_in_main(d1_idx,d2_idx) = P_invasion_in_C2(d1_idx,d2_idx);
                    elseif C1C2_coex_matrix(d1_idx,d2_idx) == 3
                        P_invasion_in_main(d1_idx,d2_idx) = P_invasion_in_C1C2(d1_idx,d2_idx);
                    end
                end
            end
        end
        %coex_matrix = C1_invasion + C2_invasion;
        
        save(filename,...
            'C1_invasion','C2_invasion','P_invasion_in_C1',...
            'P_invasion_in_C2','P_invasion_in_C1C2','P_invasion_in_main')
        fprintf("File %s saved successfully.",filename)
    
    %{
        if strcmp(C2_model,'linearized')
            if hP == 0
                % lin lin lin
                filename_matrix = sprintf('./matrices/model_linlinlin/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
                    num2str(a2),...
                    num2str(h2), num2str(aP), num2str(hP), num2str(dP));
            else
                % lin lin sat
                filename_matrix = sprintf('./matrices/model_linlinsat/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
                    num2str(a2),...
                    num2str(h2), num2str(aP), num2str(hP), num2str(dP));
            end
        else
            if hP == 0
                %lin sat lin or lin sat but that will only be visible from 
                %aP = 0
                filename_matrix = sprintf('./matrices/model_linsatlin/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
                    num2str(a2),...
                    num2str(h2), num2str(aP), num2str(hP), num2str(dP));
            else
                %lin sat sat
                filename_matrix = sprintf('./matrices/model_linsatsat/a2=%s_h2=%s_aP=%s_hP=%s_dP=%s.mat',...
                    num2str(a2),...
                    num2str(h2), num2str(aP), num2str(hP), num2str(dP));
            end
        end
    
        figure()
        subplot(1,2,1)
        imagesc(coex_matrix)
        colorbar()
        subplot(1,2,2)
        imagesc(coex_matrix+matrix_for_P)
        colorbar()
        coex_matrix = coex_matrix + matrix_for_P;
        
        matrix_struct_to_be_saved = struct("coexistence", coex_matrix);
        save(filename_matrix, '-fromstruct', matrix_struct_to_be_saved)
    %}

    end
end

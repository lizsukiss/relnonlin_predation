%%%%%                            grid making                          %%%%%
function grid = get_grid(a, h, resolution)
    % Creates numerical axis for the mortality grid
        
    maxd = a / (1 + h * a);
    grid = linspace(0, maxd, resolution + 2);
    grid = grid(2:end-1);
end

%% Geostatistics - Ordinary Kriging (i_love_kriging.m)
% By Felix Leidinger & Marina Bortoli
% January-20-2019
%
%%
clc; close all; clear all;

%% Settings
op_mode = 1; % 1: interpolation between 400x400 virtual ks observations
% 2: verification of functionality by loading a virtual reality dataset,
% drawing random samples from it, interpolation between the random samples
% and comparison with the original virtual reality
% else: abort
%% Advanced Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if op_mode == 2
    dim = 30; % Positive integer for vector / matrix dimensions --> This value
    % controls the number of observations in case of verification with 
    % virtual realities: length(x_obs) = dim^2. This assures a valid
    % sample_number.
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load the data from the virtual observations of ks (400x400)
data = load('measurement_tb.dat');

%% Load the entire observation grid (10000x10000)
grid = load('real_area.dat');

%% Load virtual reality dataset for verification
if op_mode == 2
    virtual_reality = load('ks_gaus_0,5_1,5_100.dat');
    sample_number = dim^2; % must be a positive integer < length(virtual_reality) 
    % and with an even sqrt. Larger numbers give better results for
    % interpolation but significantly increase computation time for kriging

    % Draw some random samples from virtual reality
    random_index = round(rand(sample_number,1) * length(virtual_reality));
    random_index(random_index == 0) = 1; % Assures an index > 0
    random_sample = nan(sample_number,3);
    for i = 1:length(random_index)
        random_sample(i,:) = virtual_reality(random_index(i),:);
    end
end

%% Load the variogram details - nugget, sill, range, type
variogram = load('variogram_details.dat');

nugget = variogram(1,1);
sill = variogram(1,2);
range = variogram(1,3);
type = variogram(1,4);

%% For verification with virtual reality dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if op_mode == 2
    type = 2;
    data = random_sample;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Define coordinates
x_grid = grid(:,1); % Vector of all x-coordinates of the grid
y_grid = grid(:,2); % Vector of all y-coordinates of the grid

z_sim = zeros(length(x_grid),1); % Initialize vector of simulated measurements

x_obs = data(:,1); % Vector of x-coordinates of virtual measurements
y_obs = data(:,2); % Vector of y-coordinates of virtual measurements
z_obs = data(:,3); % Vector of values of virtual measurements

sums = zeros(length(x_grid),1); % Initialize vector of lambda-sums for verification of results later on

%% Big loop for kriging
for i = 1:length(x_grid)
    % Show progress
    if mod(i,100) == 0
        clc;
        percent = 100 * (i / length(x_grid));
        disp(['progress: ',num2str(percent),' %']); 
    end
    
    % Compute distances between the virtual observations to the surrounding
    % grid points
    h_grid = zeros(length(x_obs),1);
    close_points = - ones(length(x_obs),1); % For better performance,
    % preallocate in each i-th round with -1 to the maximum size
    for j = 1:length(x_obs)    
        h_grid(j) = sqrt( (x_obs(j) - x_grid(i))^2 + (y_obs(j) - y_grid(i))^2 );
        if h_grid(j) < range
            close_points(j) = j; % Save the current position
        end
    end
    close_points = close_points(close_points ~= -1); % Shrink the array by
    % removing all elements that still have the initial (unvalid) value
    
    % Compute distances between all virtual measurements within the range
    h_obs = zeros(length(close_points),length(close_points)); % Initialize 
    % quadratic matrix of lag distances
    for k = 2:length(close_points)
       for l = 1:k-1
           h_obs(k,l) = sqrt( (x_obs(close_points(k)) - x_obs(close_points(l)))^2 + (y_obs(close_points(k)) - y_obs(close_points(l)))^2 );
           h_obs(l,k) = h_obs(k,l);
       end
    end
    
    % A * weights = b
    % --> weights = A^-1 * b
    % A represents semivariances between the i-th and j-th sampling points
    % (between the virtual measurements)
    % weights represents the vector of coefficients (lambda_i;mu), sum(weights) =! 1
    % b represents the vector of semivariances between each sampling point
    % (virtual measurement) and the target point (point on the grid)
    
    A = zeros(length(close_points)+1,length(close_points)+1); % Initialize quadratic matrix with 0
    A((length(close_points)+1),:) = 1; % Set last row to 1
    A(:,(length(close_points)+1)) = 1; % Set last column to 1
    A(length(close_points)+1,length(close_points)+1) = 0; % Set last element to 0
    
    b = zeros((length(close_points)+1),1); % Initialize vector with 0
    b(length(close_points)+1) = 1; % Set last element to 1
    
    % Fit respective variogram model
    for m = 2:length(close_points)
        switch type
            case 1 % Exponential variogram model (type == 1)
                for n = 1:m-1
                   A(m,n) = -(nugget + sill * (1 - exp(-h_obs(m,n) / range)));
                   A(n,m) = A(m,n);
                end
                for o = 1:length(close_points)
                    b(o) = -(nugget + sill * (1 - exp(-h_grid(close_points(o)) / range)));
                end
            case 2 % Gaussian variogram model (type == 2)
                for n = 1:m-1
                   A(m,n) = -(nugget + sill * (1 - exp((-h_obs(m,n)^2) / (range^2))));
                   A(n,m) = A(m,n);
                end
                for o = 1:length(close_points)
                    b(o) = -(nugget + sill * (1 - exp((-h_grid(close_points(o))^2) / (range^2))));
                end
            case 3 % Sperical variogram model (type == 3)
                for n = 1:m-1
                   A(m,n) = -(nugget + sill * (3.5 * (h_obs(m,n) / range) - 0.5 * (h_obs(m,n) / range)^3));
                   A(n,m) = A(m,n);
                end
                for o = 1:length(close_points)
                    b(o) = -(nugget + sill * (3.5 * (h_grid(close_points(o)) / range) - 0.5 * (h_grid(close_points(o)) / range)^3));
                end
            otherwise
                disp('Invalid variogram model');
        end
    end
    
    % Solve the kriging equaltion system
    weights = A \ b;
    
    lambda = zeros(1,length(x_obs));
    for p = 1:length(close_points)
        lambda(close_points(p)) = weights(p);
    end
    
    % Compute the simulated values for z
    z_sim(i) = lambda * z_obs;
    sums(i) = sum(lambda);
    
end

%% Validation of results
check = find(sums < 1 & sums > 1);
if check ~= []
    disp('error: sum(lambda) ~= 1');
else
    disp('valid: sum(lambda) == 1');
end

%% Rearrange vectors of coordinates & virtual observations to quadratic matrices for visualization
dim2 = sqrt(length(z_sim)); % Dimensions of matrix for interpolation

if op_mode == 2
    x_virtual_reality = nan(dim2,dim2);
    y_virtual_reality = nan(dim2,dim2);
    z_virtual_reality = nan(dim2,dim2);
end
x_interpolation = nan(dim2,dim2);
y_interpolation = nan(dim2,dim2);
z_interpolation = nan(dim2,dim2);

for i = 0:(dim2 - 1)
    if op_mode == 2
        x_virtual_reality(i+1,:) = virtual_reality(i*dim2+1:i*dim2+dim2,1);
        y_virtual_reality(i+1,:) = virtual_reality(i*dim2+1:i*dim2+dim2,2);
        z_virtual_reality(i+1,:) = virtual_reality(i*dim2+1:i*dim2+dim2,3);
    end
    x_interpolation(i+1,:) = x_grid(i*dim2+1:i*dim2+dim2);
    y_interpolation(i+1,:) = y_grid(i*dim2+1:i*dim2+dim2);
    z_interpolation(i+1,:) = z_sim(i*dim2+1:i*dim2+dim2);
end

%% Visualization of results
figure;
pcolor(x_interpolation,y_interpolation,z_interpolation);
ylabel('y [m]','fontsize',16);
xlabel('x [m]','fontsize',16);
set(gca,'fontsize',16,'linewidth',2);
title('log(ks) (Interpolation)','fontsize',16);
shading flat;
h=colorbar;
set(h,'fontsize',16,'linewidth',2);
hold on;
pl1 = plot(x_obs,y_obs,'k+');
if op_mode == 1
    legend(pl1,'Sampling locations','Location','southoutside');
else
    legend(pl1,'Random samples from virtual reality','Location','southoutside');
end

%% For verification with virtual reality dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if op_mode == 2
    figure;
    subplot(1,2,1);
    pcolor(x_virtual_reality,y_virtual_reality,z_virtual_reality);
    ylabel('y [m]','fontsize',16);
    xlabel('x [m]','fontsize',16);
    set(gca,'fontsize',16,'linewidth',2);
    title('log(ks) (Virtual reality)','fontsize',16);
    %shading flat;
    h=colorbar;
    set(h,'fontsize',16,'linewidth',2);
    hold on;
    %plot(x_obs,y_obs,'k+');
    subplot(1,2,2);
    pcolor(x_interpolation,y_interpolation,z_interpolation);
    ylabel('y [m]','fontsize',16);
    xlabel('x [m]','fontsize',16);
    set(gca,'fontsize',16,'linewidth',2);
    title('log(ks) (Interpolation)','fontsize',16);
    %shading flat;
    h=colorbar;
    set(h,'fontsize',16,'linewidth',2);
    hold on;
    %plot(x_obs,y_obs,'k+');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% EOF
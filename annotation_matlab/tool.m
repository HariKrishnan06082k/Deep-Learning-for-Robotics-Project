clear all;
close all;
clc;

load('mat\pc_array.mat')

%% Read data from the workspace
data = skillet;
%% Rotate if required
angle_degrees = -20;
angle_radians = angle_degrees * pi / 180;

% Define the rotation matrix around the z-axis
Rz = [cos(angle_radians) -sin(angle_radians) 0;
      sin(angle_radians) cos(angle_radians)  0;
      0                 0                  1];

% Load your Nx3 array

% Apply the rotation matrix
data = data * Rz;
%% Start Labeling
% Initialize variables for storing ROIs and labels
ROIs = {};
labels = {};
% Ask user how many classes for labels
object_name = input('Enter name of the object: ','s');
n_class = input('Enter number of class labels: ');
% For every n_class do the following:
for i=1:n_class
    % Plot data
    figure;
    scatter3(data(:,1), data(:,2), data(:,3), '.');
    title('Interactive Scatter Plot');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    xlim([-1 1])
    ylim([-1 1])
    zlim([-1 1])
    % Enable zooming
    zoom on;
    % Draw cuboid in interactive plot and get roi
    disp('Draw cuboid in the plot and double-click when done...');
    roi = drawcuboid;   
    pause;
    
    % Get points of the cuboid
    minPt = roi.Position(:,1:3);
    maxPt = roi.Position(:,4:6);
    
    indices = data(:,1) >= minPt(1) & data(:,2) >= minPt(2) & data(:,3) >= minPt(3) & ...
              data(:,1) <= maxPt(1) & data(:,2) <= maxPt(2) & data(:,3) <= maxPt(3);
    
    % Store the points within the cuboid
    points_within_cuboid = data(indices,:);
    
    ROI_current = points_within_cuboid;
    % Plot the current ROI points in new figure and ask user if this is okay
    figure;
    scatter3(ROI_current(:,1), ROI_current(:,2), ROI_current(:,3), '.','red');
    title('Interactive Scatter Plot');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    xlim([-1 1])
    ylim([-1 1])
    zlim([-1 1])
    % Ask user for inputing class variable:
    label = input('Class label: ','s');
    
    % Ask user of the user is satisfied with the segmented ROI
    satisfied = input('Continue with remaining point cloud? (Y / N) : ','s');
    % If YES, then store the remaining point cloud from original data for ROI in next loop:
    % If NOT, then redo this iteration again
    
    if (strcmpi(satisfied, 'Y'))
        % Append ROI and label to respective cell arrays
        ROIs = [ROIs; points_within_cuboid];
        labels = [labels; repmat(label, size(points_within_cuboid,1),1)];
        % Extract remaining clouds:
        [~, idx] = ismember(data, points_within_cuboid, 'rows');
        unlabeled_idx = find(idx == 0);
        % Store unlabeled points in a new variable
        data = data(unlabeled_idx, :); % Data for next iteration
        % Close the figure
        close
    else
        % If user is not satisfied, repeat the same iteration again
        i = i-1;
        % Close the figure
        close
    end
end

% Combine all ROIs and labels to make final point cloud and label structure
pc_comb = vertcat(ROIs{:});
label_comb = [];
for i = 1:length(labels)
    label_comb = [label_comb;string(labels{i})];
end
%label_comb = vertcat(labels{:});
pc_labelled = struct('xyz', pc_comb, 'labels', label_comb);


pc_table = struct2table(pc_labelled);
pc_table.Properties.VariableNames = {'Coord', 'Labels'};

pc_labelled_ordered = pointCloud(pc_labelled.xyz); 
%% Augment

% Define random rigid transformations
numTransforms = 5;
transformations = cell(numTransforms, 1);
for i = 1:numTransforms
    % Generate random translation and rotation vectors
    translation = rand(1,3);
    rotation = rand(1,3);
    
    % Create transformation matrix
    R = axang2rotm([rotation, norm(rotation)]);
    T = [R, translation'; 0 0 0 1];
    T(:,4) = [0; 0; 0; 1];
    % Save transformation matrix
    transformations{i} = affine3d(T);
end

% Apply random rigid transformations
pointCloudList = [];
for i = 1:numTransforms
    pointCloudList= [pointCloudList;pctransform(pc_labelled_ordered, transformations{i})];
end
% Iterate over each point cloud in the list and add jittering
jitterAmount = 0.001;
for i = 1:length(pointCloudList)
    % Get the current point cloud
    ptCloud_i = pointCloudList(i);
    
    % Jitter the point cloud coordinatesg
    jitteredCoords_i = ptCloud_i.Location + jitterAmount*randn(size(ptCloud_i.Location));

    % Create a new point cloud object with jittered coordinates
    jitteredPtCloud = pointCloud(jitteredCoords_i, 'Color', ptCloud_i.Color);
    
    % Update the current point cloud in the list with the jittered point cloud
    pointCloudList= [pointCloudList;jitteredPtCloud];
end

%% Save list of pointcloud as csv
% Loop through each pointcloud in list, add label from structure
% Save it


for i = 1:length(pointCloudList)
    coord_i = pointCloudList(i).Location;
    label_i = pc_labelled.labels;
    pc_labelled_i = struct('xyz', coord_i, 'labels', label_i);
    pc_table_i = struct2table(pc_labelled_i);
    pc_table_i.Properties.VariableNames = {'Coord', 'Labels'};
    
    % Save this table as csv
    file_name = object_name+"_"+string(i);
    writetable(pc_table_i, "csv\"+object_name+"\"+string(file_name)+ ".csv")
    % Save the struct to a mat-file
    save("mat\"+object_name+"\"+ string(file_name)+ ".mat", '-struct', 'pc_labelled_i');
end

% 
% % Name the csv file according to the name of the initial data variable :
% % here hammer
% file_name = input('File name: ');
% writetable(pc_table, "csv\"+string(file_name)+ ".csv")
% % Save the struct to a mat-file

% save("mat\"+ string(file_name)+ ".mat", pc_labelled);

%clear, close all, clc;
function compressive_sensing_radar_v1(scene,meas) 
%myDir = char(strcat('/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1/scene', string(scene),'/radar'));
%saveDir = char(strcat('/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1/scene', string(scene),'/radar-recons-sparse-40-', meas));

scenes={'city_3_7','night_1_4', 'motorway_2_2','snow_1_0','tiny_foggy'}

for scene_num=1:length(scenes)
	scene = scenes{scene_num}

myDir= char(strcat('/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/',scene,'/Navtech_Polar'))
saveDir=char(strcat('/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/',scene,'/reconstruct-polar-same-meas-55'))


myFiles = dir(fullfile(myDir,'*.png'));
meas
saveDir
parallel = 19; length(myFiles);%12;
outer_loop = [2:parallel:19];%length(myFiles)];

if ~exist(saveDir, 'dir')
       mkdir(saveDir)
    end

for p = 1:length(outer_loop)
recons_full = [];
snrs_full = [];
MAE_full = [];
saveFileNameFull = [];
%if p == length(outer_loop)
%    parallel = 2;
%end
%parallel = 1;


for k = 1:parallel %%%%%%%%
    current = outer_loop(p) + (k-1)
    baseFileName = myFiles(current).name;
    fullFileName = fullfile(myDir, baseFileName);
    disp(fullFileName)
    A = imread(fullFileName);
    %baseFileName = strrep(baseFileName, '.png', '.mat');
    %meta = A(:, 1:11); 
    %A = A(:,12:3779);

    A = A.';
    [height, width] = size(A);
    %A = A([1:50],[12:87]);
    w = 20;
    h = 48; %112 - 12; %87 - 12;
    rate = double(int16((w*h)*0.55));


    %Phi = randn(floor(0.1*50*50),50*50);

    %%%%
    I = eye(w*h);
    I = I(1:floor(0.55*w*h),1:w*h);
    cols = size(I,2);
    P = randperm(cols);
    Phi = I(:,P);
    %%%%


    snrs = [];
    MAEs = [];
    rows = [1: w: 401]
    columns = [1: h: 577]
    final_A = [];
    final_rate = 0;
    for c = 1:length(rows)-1
        c,rate;
        final_A_column = [];
        for d = 1:length(columns)-1
	    final_rate = final_rate + rate;
            %final_rate, rate
            A_ = A([rows(c):rows(c+1)-1],[columns(d):columns(d+1)-1]);
	    %continue %%%%%%%%%%
            %x1 = compressed_sensing_example(A_, w, h, rate, meas);
	    x1 = compressed_sensing_example_parallel(A_, w, h, rate,Phi); %%%%
            x1 = uint8(x1);
            peak = psnr(A_,x1);
            snrs = [snrs;peak];
            MAE=sum(abs(A_(:)-x1(:)))/numel(A_);
            MAEs = [MAEs; MAE];
            final_A_column = horzcat(final_A_column, x1);
            
        end
        final_A = vertcat(final_A, final_A_column);        
    end
    final_rate
    %continue %%%%%%%%%%
    %final_A_meta = final_A; %horzcat(meta, final_A);
    %recons_full = [recons_full ; final_A_meta];
    %snrs_full = [snrs_full; snrs];
    %MAE_full = [MAE_full; MAEs];
    final_A = final_A.';
    [recons_h, recons_w] = size(final_A);
    final_A_reshaped = zeros(576,400);
    recons_h
    recons_w
    final_A_reshaped(1:recons_h,1:recons_w) = final_A;
    final_A_reshaped = uint8(final_A_reshaped);
    fullFileNameRecons = fullfile(saveDir, baseFileName);
    imwrite(final_A_reshaped,fullFileNameRecons);
    baseFileName = strrep(baseFileName, '.png', '.mat');
    fullFileNameRecons = fullfile(saveDir, baseFileName);
    save(fullFileNameRecons, 'snrs', 'MAEs')
    mean(snrs)
    mean(MAEs)

    %saveFileNameFull = [saveFileNameFull; fullFileNameRecons];
end
for n = 1:parallel
continue
final_A_meta = recons_full((n-1)*1152 + 1: n*1152, :);
snrs = snrs_full((n-1)*529 +1 : n*529);
MAEs = MAE_full((n-1)*529 +1 : n*529);
fullFileNameRecons = saveFileNameFull(n,:);
imwrite(final_A,fullFileNameRecons);
mean(snrs)
mean(MAEs)
%save(fullFileNameRecons, 'final_A_meta', 'snrs', 'MAEs')
end
end
end

end

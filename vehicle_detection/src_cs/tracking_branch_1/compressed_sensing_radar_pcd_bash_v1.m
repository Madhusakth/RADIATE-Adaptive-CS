%clear, close all, clc;

function compressed_sensing_radar_pcd_bash(k)

    myDir = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/city_3_7';
    radarDir = char(strcat(myDir,'/Navtech_Cartesian/'));
    saveDir= char(strcat(myDir,'/30-final-rad-info-same-meas/'));
    point_radDir = char(strcat(myDir,'/30-net_output_idx-same-meas/'));

    if ~exist(saveDir, 'dir')
       mkdir(saveDir)
       end

    myFiles = dir(fullfile(radarDir,'*.png'));
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(radarDir, baseFileName);
    %baseFileName = strrep(baseFileName, '.png', '.mat');
    

    obj_file = myFiles(k-1).name;
    obj_file;
    pcd_file = char(strcat(int2str(k-1),'_'))
    pcd_row_file = strcat(point_radDir, pcd_file, 'row');
    pcd_row = load(pcd_row_file);
    pcd_rows = pcd_row.obj_rows;
    pcd_rows = pcd_rows + 1;
    
    
    pcd_column_file = strcat(point_radDir, pcd_file, 'column');
    pcd_column = load(pcd_column_file);
    pcd_columns = pcd_column.obj_columns;
    pcd_columns = pcd_columns + 1;

    length(pcd_rows)
    length(pcd_columns)
    
    disp(fullFileName)
    
    A = imread(fullFileName);
    [height, width] = size(A);
    w = 50;
    h = 50
    snrs = [];
    MAEs = [];
    rows = [1: w: 1152];
    columns = [1: h: 1152];
    
    Imp = length(pcd_rows)

    Other = 529 - Imp
    A1 = optimvar('A1','LowerBound', 0.3, 'UpperBound',0.4);
    B1 = optimvar('B1','LowerBound',0.10 ,'UpperBound' ,0.3);
    prob = optimproblem('Objective' , Imp*50*50*A1 + Other*50*50*B1 ,'ObjectiveSense','max');
    prob.Constraints.c1 = Imp*50*50*A1 + Other*50*50*B1 <= 396750;
    prob.Constraints.c2 = A1 >= 1.2*B1;
    %prob.Constraints.c3 = Imp*50*50*A1 + Other*50*50*B1 >= 132000;

    problem = prob2struct(prob);
    [sol,fval,exitflag,output] = linprog(problem);
    sol
    Imp*floor(50*50*sol(1)) + Other*floor(50*50*sol(2))
    
    %%%%
    I = eye(50*50);
    I = I(1:floor(sol(1)*50*50),1:50*50);
    cols = size(I,2);
    P = randperm(cols);
    Phi_1 = I(:,P);

    I = eye(50*50);
    I = I(1:floor(sol(2)*50*50),1:50*50);
    cols = size(I,2);
    P = randperm(cols);
    Phi_2 = I(:,P);
    %%%%

    final_A = [];
    final_rate = 0;
    for c = 1:23
        c
        final_A_column = [];
        for d = 1:23
	    flag = 0;
	    for q=1:length(pcd_rows)
                if pcd_rows(q) == c && pcd_columns(q)==d
                     rate = floor(sol(1)*w*h);
		     Phi = Phi_1; %%%%
		     flag=1;
		end
            end
            if flag==0;
	       rate = floor(sol(2)*w*h);
	       Phi = Phi_2; %%%%
	    end


            final_rate  = final_rate + rate;
            %continue %%%%%%%%%%%%
            %c,d,rate
            A_ = A([rows(c):rows(c+1)-1],[columns(d):columns(d+1)-1]);
            x1 = compressed_sensing_example_parallel(A_, w, h, rate,Phi); %%%%
            x1 = uint8(x1);
            peak = psnr(A_,x1);
            snrs = [snrs;peak];
            MAE=sum(abs(A_(:)-x1(:)))/numel(A_);
            MAEs = [MAEs;MAE];
            final_A_column = horzcat(final_A_column, x1);
        end
        final_A = vertcat(final_A, final_A_column);
        row_samples = [];
    end
    final_rate
    %exit %%%%%%%%%%%%
    final_A_reshaped = zeros(1152,1152);
    final_A_reshaped(1:1150,1:1150) = final_A;
    final_A_reshaped = uint8(final_A_reshaped);

    
    fullFileNameRecons = fullfile(saveDir, baseFileName);
    imwrite(final_A_reshaped,fullFileNameRecons);
    
    baseFileName = strrep(baseFileName, '.png', '.mat');
    fullFileNameRecons = fullfile(saveDir, baseFileName);
    save(fullFileNameRecons, 'snrs', 'MAEs')
exit
end

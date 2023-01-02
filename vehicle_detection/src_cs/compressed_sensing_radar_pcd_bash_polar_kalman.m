%clear, close all, clc;

function compressed_sensing_radar_pcd_bash_polar(k,scene,radar_output,sr)

    myDir = char(strcat('../../data/radiate/',scene));
    radarDir = char(strcat(myDir,'/Navtech_Polar/'));
    saveDir= char(strcat(myDir,'/',radar_output))
    
    if sr==10
    	point_radDir = char(strcat(myDir,'/10-net_output_idx-polar-kalman/'))
	samp_rate=0.1
    elseif sr==20
    	point_radDir = char(strcat(myDir,'/20-net_output_idx-polar-kalman/'))
	samp_rate=0.2
    else
	point_radDir = char(strcat(myDir,'/30-net_output_idx-polar-kalman/'))    
	samp_rate=sr*0.01
    end

    if ~exist(saveDir, 'dir')
       mkdir(saveDir)
       end

    myFiles = dir(fullfile(radarDir,'*.png'));
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(radarDir, baseFileName);

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
    A = A.';
    [height, width] = size(A);
    w = 20;
    h = 48;
    snrs = [];
    MAEs = [];
    rows = [1: w: 401];
    columns = [1: h: 577];

    uniform = 0;
    if length(pcd_rows) == 0
	%samp_rate = 0.8 %%%%%%%%%%%%%%%%%%
        I = eye(w*h);
        I = I(1:floor(samp_rate*w*h),1:w*h);
        cols = size(I,2);
        P = randperm(cols);
        Phi = I(:,P);
        uniform = 1;
	rate = samp_rate*w*h;
        %disp('************ Uniform sampling ************ override 80% *************');
    end
    if uniform == 0
    
    	Imp = length(pcd_rows)
	disp('calculate LP')

    	Other = 240 - Imp


	if sr==10
		disp('10% LP conditions')
    		A1 = optimvar('A1','LowerBound', 0.1, 'UpperBound',0.55);
    		B1 = optimvar('B1','LowerBound',0.07 ,'UpperBound' ,0.1);
    		prob = optimproblem('Objective' , Imp*w*h*A1 + Other*w*h*B1 ,'ObjectiveSense','max');
    		prob.Constraints.c1 = Imp*w*h*A1 + Other*w*h*B1 <= 23040;
    		prob.Constraints.c2 = A1 >= 1.1*B1;

	elseif sr==20
		disp('20% LP conditions')
		A1 = optimvar('A1','LowerBound', 0.2, 'UpperBound',0.55);
                B1 = optimvar('B1','LowerBound',0.07 ,'UpperBound' ,0.2); 
                prob = optimproblem('Objective' , Imp*w*h*A1 + Other*w*h*B1 ,'ObjectiveSense','max');
                prob.Constraints.c1 = Imp*w*h*A1 + Other*w*h*B1 <= 46080;
                prob.Constraints.c2 = A1 >= 1.1*B1;
	else
		disp('LP conditions')
		disp(samp_rate)
		A1 = optimvar('A1','LowerBound', samp_rate, 'UpperBound',0.55);
                B1 = optimvar('B1','LowerBound',0.07 ,'UpperBound' ,samp_rate);
                prob = optimproblem('Objective' , Imp*w*h*A1 + Other*w*h*B1 ,'ObjectiveSense','max');
                prob.Constraints.c1 = Imp*w*h*A1 + Other*w*h*B1 <= samp_rate*400*576;
                prob.Constraints.c2 = A1 >= 1.1*B1;
	end

    	problem = prob2struct(prob);
    	[sol,fval,exitflag,output] = linprog(problem);
    	sol
    	Imp*floor(w*h*sol(1)) + Other*floor(w*h*sol(2))
    
    	%%%%
    	I = eye(w*h);
    	I = I(1:floor(sol(1)*w*h),1:w*h);
    	cols = size(I,2);
    	P = randperm(cols);
    	Phi_1 = I(:,P);

    	I = eye(w*h);
    	I = I(1:floor(sol(2)*w*h),1:w*h);
    	cols = size(I,2);
    	P = randperm(cols);
    	Phi_2 = I(:,P);
    	%%%%

    end

    final_A = [];
    final_rate = 0;
    for c = 1:length(rows)-1
        c
        final_A_column = [];
        for d = 1:length(columns)-1

	  if uniform == 0
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
          end

            final_rate  = final_rate + rate;
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
exit
end

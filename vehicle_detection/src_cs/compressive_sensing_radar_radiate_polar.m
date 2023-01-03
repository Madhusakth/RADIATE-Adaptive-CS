%clear, close all, clc;
function compressive_sensing_radar_radiate_polar() 

scenes={'city_3_7','night_1_4', 'motorway_2_2','snow_1_0','fog_6_0'}
%scenes={'fog_6_0'}

samp_rate = 0.20;
meas = 'BPD';

for scene_num=1:length(scenes)
	scene = scenes{scene_num}

	myDir= char(strcat('../../data/radiate/',scene,'/Navtech_Polar'))
	saveDir=char(strcat('../../data/radiate/',scene,'/standard-cs-BPD-40-img-20-1'))


	myFiles = dir(fullfile(myDir,'*.png'));
	parallel = length(myFiles);

	if ~exist(saveDir, 'dir')
	       mkdir(saveDir)
	    end


	for current = 1:41 %40 %parallel
	    baseFileName = myFiles(current).name;
	    fullFileName = fullfile(myDir, baseFileName);
	    disp(fullFileName)
	    A = imread(fullFileName);

	    A = A.';
	    [height, width] = size(A);
	    w = 20;
	    h = 48; 
	    rate = double(int16((w*h)*samp_rate));



	    if strcmp(meas,'gauss')
		
		Phi = randn(floor(samp_rate*w*h),w*h);
		if current == 1
		   disp('gauss matrix');
		   size(Phi)
		end

	    elseif strcmp(meas,'BPD')
		I = eye(w*h);
		I = I(1:floor(samp_rate*w*h),1:w*h);
		cols = size(I,2);
		P = randperm(cols);
		Phi = I(:,P);
		disp('BPD matrix');

	    else
		rate = samp_rate*w*h;
		n = w*h;
		Phi = zeros(rate,n);
		num = floor(n/rate);
		for i = 1:rate
			Phi(i, (i-1)*num+1: (i-1)*num + num) = 1;
		end
		cols = size(Phi,2);
		P = randperm(cols);
		Phi = Phi(:,P);
		disp('BPBD matrix');
	    end
		
	    %I = eye(w*h);
	    %I = I(1:floor(samp_rate*w*h),1:w*h);
	    %cols = size(I,2);
	    %P = randperm(cols);
	    %Phi = I(:,P);

	    snrs = [];
	    MAEs = [];
	    rows = [1: w: 401];
	    columns = [1: h: 577];
	    final_A = [];
	    final_rate = 0;
	    for c = 1:length(rows)-1
		c,rate;
		final_A_column = [];
		for d = 1:length(columns)-1
		    final_rate = final_rate + rate;
		    A_ = A([rows(c):rows(c+1)-1],[columns(d):columns(d+1)-1]);
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

	end
end

end

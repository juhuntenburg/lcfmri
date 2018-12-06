close all
clear all

addpath(genpath('/home/julia/workspace/lcfmri/matlab/'));

data_dir='/home/julia/projects/lc/raw/visual_clem/4/';
scan='2';
acq_params=readBrukerParamFile(strcat(data_dir,'acqp'));
method_params=readBrukerParamFile(strcat(data_dir,'method'));
visu_params=readBrukerParamFile(strcat(data_dir, filesep, 'pdata', filesep, scan, filesep, 'visu_pars'));
[data, visu] = readBruker2dseq(strcat(data_dir, filesep, 'pdata', filesep, scan, filesep,'2dseq'), visu_params);
data=squeeze(data);

data_added = sum(abs(data), 5);
data_sos = sqrt(sum(abs(data).^2, 5));

save('/home/julia/projects/lc/raw/visual_clem/4/data_added.mat','data_added');
save('/home/julia/projects/lc/raw/visual_clem/4/data_sos.mat','data_sos');
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

data_combined = sqrt(mean(abs(data),5));
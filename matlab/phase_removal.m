clear all
close all

addpath(genpath('/home/julia/workspace/lcfmri/matlab/'));

% load data
data_dir='/home/julia/projects/lc/raw/20181006_165517_JH_LC_rsfMRI_03_1_1/25/';
scan='2';
acq_params=readBrukerParamFile(strcat(data_dir,'acqp'));
method_params=readBrukerParamFile(strcat(data_dir,'method'));
visu_params=readBrukerParamFile(strcat(data_dir, filesep, 'pdata', filesep, scan, filesep, 'visu_pars'));
[complex, visu] = readBruker2dseq(strcat(data_dir, filesep, 'pdata', filesep, scan, filesep,'2dseq'), visu_params);

denoised = zeros(size(complex));
[nx,ny,nz,nt] = size(complex); 

tic
 for cc=1:nz,
   for kk=1:nt 
       
       a = squeeze(complex(:,:,cc,kk));
       u = tvdenoise(a,15,'L1');
       b = angle(u);
       
       denoised(:,:,cc,kk) = (a.*(exp(-1i*b))); 
   end
 end
toc

denoised = squeeze(denoised);
real_data = real(denoised);
imag_data = imag(denoised);

save('/home/julia/projects/lc/processed/phase_corrected.mat')
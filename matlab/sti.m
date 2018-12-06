
%addpath
addpath(genpath('/home/shemesh/julia/'));

% load data
source='/home/shemesh/julia/data/16/pdata/5/';
raw='/home/shemesh/julia/data/16/';
acq_params=readBrukerParamFile(strcat(raw,'acqp'));
method_params=readBrukerParamFile(strcat(raw,'method'));
visu_params=readBrukerParamFile(strcat(source, 'visu_pars'));
[image, visu] = readBruker2dseq(strcat(source,'2dseq'), visu_params);
load('/home/shemesh/julia/data/16_mask.mat');
mask=double(data);

% compute raw phase image from complex data
phase=angle(image);
magnitude=abs(image);
save('/home/shemesh/julia/data/16_phase.mat', 'phase');
save('/home/shemesh/julia/data/16_magnitude.mat', 'magnitude');

% real_mat=real(image);
% imaginary_mat=imag(image);

% compute laplacian
[unwrapped_phase, laplacian]=MRPhaseUnwrap(phase_mat, 'voxelsize', [0.19, 0.16, 0.243], 'padsize', [12,12,12]);
save('/home/shemesh/julia/data/16_unwrapped_phase.mat', 'unwrapped_phase');

% combined phase unwrapping and background removal
[tissue_phase]=iHARPERELLA(laplacian,mask,'voxelsize',[0.19, 0.16, 0.24],'niter',100);
save('/home/shemesh/julia/data/16_tissue_phase.mat', 'tissue_phase');

% reconstruct complex image
% complex = magnitude.*exp(i*tissue_phase);
% mag_clean = abs(complex);
% save('/home/shemesh/julia/data/16_cleaned_complex.mat', 'complex');
% save('/home/shemesh/julia/data/16_cleaned_magnitude.mat', 'mag_clean');
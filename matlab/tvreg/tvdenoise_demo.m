% TV denoising demo using tvrestore, Pascal Getreuer 2010
%
% This is a MATLAB script.  To run it, enter on the MATLAB console
%   >> tvdenoise_demo


NoiseLevel = 0.06; 
lambda = 15;

% Simulate a noisy image
uexact = double(imread('einstein.png'))/255;
f = uexact + randn(size(uexact))*NoiseLevel;

% Make a figure 
clf;
set(gcf,'Color',[1,1,1],'NumberTitle','off','Name','TV Denoising');
compareimages(f,'Input',f,'Denoised');
shg;

% Denoise
u = tvdenoise(f,lambda,[],[],[],@tvregsimpleplot);

compareimages(f,'Input',u,'Denoised');

#3dAutomask -prefix mask.nii.gz data_sos.nii.gz
import nibabel as nb
import numpy as np
from nipype.algorithms.confounds import TSNR

data_dir = "/home/julia/projects/lc/nifti/visual_clem_nifti/CL181030fmrssouris3/CL181030fmrssouris3_4/"

added_file = data_dir + "data_added.nii.gz"
sos_file = data_dir + "data_sos.nii.gz"
real_file = data_dir + "data_real.nii.gz"
mask_file = data_dir + "mask.nii.gz"

# calculate tsnr
for i in ['added', 'sos', 'real']:

    data = nb.load(data_dir+"data_%s.nii.gz"%i).get_data()
    tsnr = np.nan_to_num(np.mean(data,axi=3)/np.std(data))

    tsnr = TSNR(in_file=data_dir+"data_%s.nii.gz"%i, tsnr_file=data_dir+"data_%s_tsnr.nii.gz"%i)
    tsnr.run()

mask = nb.load(mask_file).get_data()

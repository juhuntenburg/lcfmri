#!/bin/python

import os
from glob import glob
from bruker2nifti.converter import Bruker2Nifti

data_in = '/home/julia/projects/lc/raw/20181220_102400_JH_20181220_lcrsfmri_test03_1_1/'
data_out = '/home/julia/projects/lc/nifti/20181220_102400_JH_20181220_lcrsfmri_test03_1_1_nifti/'

if not os.path.isdir(data_out):
    os.mkdir(data_out)

bru = Bruker2Nifti(data_in, data_out, study_name="")
bru.get_acqp = True
bru.get_method = True
bru.get_reco = True
bru.verbose = 0
bru.convert()

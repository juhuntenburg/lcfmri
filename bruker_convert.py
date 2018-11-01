#!/bin/python

import os
from glob import glob
from bruker2nifti.converter import Bruker2Nifti

data_in = '/home/julia/projects/lc/raw/20181101_090855_JH_LC_rsfMRI_07_1_1/'
data_out = '/home/julia/projects/lc/nifti/20181101_090855_JH_LC_rsfMRI_07_1_1_nifti/'

if not os.path.isdir(data_out):
    os.mkdir(data_out)

bru = Bruker2Nifti(data_in, data_out, study_name="")
bru.get_acqp = True
bru.get_method = True
bru.get_reco = True
bru.verbose = 0
bru.convert()

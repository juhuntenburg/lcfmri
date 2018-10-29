#!/bin/python

import os
from glob import glob
from bruker2nifti.converter import Bruker2Nifti

data_in = '/home/julia/projects/lc/20181006_165517_JH_LC_rsfMRI_03_1_1/'
data_out = '/home/julia/projects/lc/20181006_165517_JH_LC_rsfMRI_03_1_1_nifti/'

if not os.path.isdir(data_out):
    os.mkdir(data_out)

bru = Bruker2Nifti(data_in, data_out, study_name="")
bru.get_acqp = True
bru.get_method = True
bru.get_reco = True
bru.verbose = 0
bru.convert()

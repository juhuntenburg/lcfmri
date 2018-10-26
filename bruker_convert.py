#!/bin/python

import os
from glob import glob
from bruker2nifti.converter import Bruker2Nifti

data_in = '/home/julia/projects/lc/sti_test_standard/'
data_out = '/home/julia/projects/lc/sti_test_standard_nifti/'

if not os.path.isdir(data_out):
    os.mkdir(data_out)

bru = Bruker2Nifti(data_in, data_out, study_name="")
bru.get_acqp = True
bru.get_method = True
bru.get_reco = True
bru.verbose = 0
#correct_slope?

#print(bru.scans_list)
#print(bru.list_new_name_each_scan)

bru.convert()

from nipype.pipeline.engine import Node, Workflow, MapNode
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.nipy as nipy
import nipype.interfaces.ants as ants
import nipype.interfaces.afni as afni
import nipype.interfaces.freesurfer as fs
import nipype.algorithms.rapidart as ra
from functions import strip_rois_func, motion_regressors, median, selectindex, nilearn_denoise, weigthed_avg, pca_denoising

dataset = 'LC_rsfMRI_03'
struct = '20'
lc = '24'
resting = ['25']
vol_to_remove = 10
motion_norm = 0.04
z_thr = 2
tr = 3
bandpass = [None, 0.01]  # rather 0.3 but too high for our temporal resolution

# directories
working_dir = '/home/julia/projects/lc/working_dir/'
data_dir= '/home/julia/projects/lc/nifti/'
out_dir = '/home/julia/projects/lc/processed/'

# main workflow
preproc = Workflow(name='preproc')
preproc.base_dir = working_dir
preproc.config['execution']['crashdump_dir'] = preproc.base_dir + "/crash_files"

# iterate over sessions
session_infosource = Node(util.IdentityInterface(fields=['session']),
                  name='session_infosource')
session_infosource.iterables=[('session', resting)]

# select files
templates = {'rest' : '*{dataset}*/*/*_{session}/*{session}.nii.gz',
             'brain' : '*{dataset}*/*/*_{struct}/*{struct}.nii.gz',
             'lc' : '*{dataset}*/*/*_{lc}/*{lc}.nii.gz',
             #'lc_mask' :
             #'lc2brain_init' :
           }
selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")
selectfiles.inputs.dataset = dataset
selectfiles.inputs.struct = struct
selectfiles.inputs.lc = lc
preproc.connect([(session_infosource, selectfiles, [('session', 'session')])])


############################
# Functional preprocessing #
############################

# TODO: Image reconstruction and background phase removal?

# Remove first volumes
remove_vol = Node(util.Function(input_names=['in_file','t_min'],
                                output_names=["out_file"],
                                function=strip_rois_func),
                  name='remove_vol')
remove_vol.inputs.t_min = vol_to_remove
preproc.connect([(selectfiles, remove_vol, [('rest', 'in_file')])])

# Thermal noise removal
func_denoise = Node(util.Function(input_names=['dwi'],
                                    output_names=['denoised_data', 'sigmas',
                                                  'preserved_components'],
                                     function=pca_denoising),
                                     name='func_denoise')
preproc.connect([(remove_vol, func_denoise, [('out_file', 'dwi')])])

# motion correction
moco = Node(nipy.SpaceTimeRealigner(),name="moco")
preproc.connect([(func_denoise, moco, [('denoised_data', 'in_file')])])

# compute median
median = Node(util.Function(input_names=['in_files'],
                       output_names=['median_file'],
                       function=median),
              name='median')

preproc.connect([(moco, median, [('out_file', 'in_files')])])

# bias field correction
biasfield = Node(ants.segmentation.N4BiasFieldCorrection(dimension=3,
                    n_iterations=[150,100,50,30], convergence_threshold=1e-11,
                    bspline_fitting_distance = 10, bspline_order = 4,
                    shrink_factor = 2,),name='biasfield')
preproc.connect([(median, biasfield, [('median_file', 'input_image')])])

# compute functional mask
func_mask = Node(afni.Automask(dilate=1, args='-peels 3',
                               outputtype='NIFTI_GZ'),
                 name='func_mask')

preproc.connect([(biasfield, func_mask, [('output_image', 'in_file')])])


# artefact detection
artefact = Node(ra.ArtifactDetect(save_plot=True,
                                  use_norm=True,
                                  parameter_source='NiPy',
                                  mask_type='file',
                                  norm_threshold=motion_norm,
                                  zintensity_threshold=z_thr,
                                  use_differences=[True,False]),
                 name='artefact')

preproc.connect([(moco, artefact, [('out_file', 'realigned_files'),
                                   ('par_file', 'realignment_parameters')]),
                 (func_mask, artefact, [('out_file', 'mask_file')]),
                 ])

# calculate motion regressors
motreg = Node(util.Function(input_names=['motion_params', 'order','derivatives'],
                            output_names=['out_files'],
                            function=motion_regressors),
                 name='motion_regressors')
motreg.inputs.order=2
motreg.inputs.derivatives=2
preproc.connect([(moco, motreg, [('par_file','motion_params')])])

# nuissance regression
regress = Node(util.Function(input_names=['in_file', 'brain_mask',
                                          'motreg_file', 'outlier_file',
                                          'bandpass', 'tr'],
                             output_names=['denoised_img', 'denoised_data',
                                           'confounds'],
                             function=nilearn_denoise), name='regress')
regress.inputs.tr = tr
regress.inputs.bandpass = bandpass

preproc.connect([(moco, regress, [('out_file', 'in_file')]),
                 (func_mask, regress, [('out_file', 'brain_mask')]),
                 (motreg, regress, [(('out_files',selectindex,[0]), 'motreg_file')]),
                 (artefact, regress, [('outlier_files', 'outlier_file')])
                 ])


############################
# Structural preprocessing #
############################

# Thermal noise removal
struct_denoise = Node(util.Function(input_names=['dwi'],
                                    output_names=['denoised_data', 'sigmas',
                                                  'preserved_components'],
                                     function=pca_denoising),
                                     name='struct_denoise')
preproc.connect([(selectfiles, struct_denoise, [('brain', 'dwi')])])

# Split structural image in individual echo times
img_split = Node(fsl.Split(dimension='t', output_type='NIFTI_GZ'),
                 name='img_split')
preproc.connect([(struct_denoise, img_split, [('denoised_data', 'in_file')])])

# Bias field correction of each echo time
struct_bias = MapNode(ants.N4BiasFieldCorrection(dimension=3,
                                                 n_iterations=[100,100,100,100],
                                                 convergence_threshold=0.0,),
                                                 iterfield=['input_image'],
                                                 name='struct_bias')

preproc.connect([(img_split, struct_bias, [('out_files', 'input_image')])])

# Merge corrected files again
img_merge = Node(fsl.Merge(dimension='t', output_type='NIFTI_GZ'),
                 name='img_merge')
preproc.connect([(struct_bias, img_merge, [('output_image','in_files')])])

# Create average across all echo times
average = Node(fsl.MeanImage(), name='struct_average')
preproc.connect([(img_merge, average, [('merged_file','in_file')])])

# Skull stripping on first echo time (highest SNR)
skullstrip = Node(afni.SkullStrip(outputtype='NIFTI_GZ',
                                  args='-rat -push_to_edge -orig_vol'),
                  name='skullstrip')
preproc.connect([(struct_bias, skullstrip, [(('output_image', selectindex, [0]),
                                              'in_file')])])

# Binarize mask
struct_mask = Node(fs.Binarize(out_type = 'nii.gz'), name='struct_mask')
preproc.connect([(skullstrip, struct_mask, [('out_file','in_file')])])

# Create masked, weighted image for coregistration
weighted_avg = Node(util.Function(input_names=['in_file', 'mask_file'],
                                    output_names=['out_file'],
                                     function=pca_denoising),
                                     name='weighted_avg')
preproc.connect([(img_merge, weighted_avg, [('merged_file', 'in_file')]),
                 (struct_mask, weighted_avg, [('binary_file', 'mask_file')])])


################
# Registration #
################

#preproc.run()

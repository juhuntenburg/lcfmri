#!/bin/python

def nilearn_denoise(in_file, brain_mask, csf_mask, artefact_ts, motreg_file, outlier_file,
                    bandpass, tr):
    """Clean time series using Nilearn high_variance_confounds to extract
    CompCor regressors and NiftiMasker for regression of all nuissance regressors,
    detrending, normalziation and bandpass filtering.
    """
    import numpy as np
    import nibabel as nb
    import os
    from nilearn.image import high_variance_confounds
    from nilearn.input_data import NiftiMasker
    from nipype.utils.filemanip import split_filename

    # reload niftis to round affines so that nilearn doesn't complain
    csf_nii=nb.Nifti1Image(nb.load(csf_mask).get_data(), np.around(nb.load(csf_mask).get_affine(), 2), nb.load(csf_mask).get_header())
    time_nii=nb.Nifti1Image(nb.load(in_file).get_data(),np.around(nb.load(in_file).get_affine(), 2), nb.load(in_file).get_header())

    # infer shape of confound array
    # not ideal
    confound_len = nb.load(in_file).get_data().shape[3]

    # create outlier regressors
    outlier_regressor = np.empty((confound_len,1))
    try:
        outlier_val = np.genfromtxt(outlier_file)
    except IOError:
        outlier_val = np.empty((0))
    for index in np.atleast_1d(outlier_val):
        outlier_vector = np.zeros((confound_len, 1))
        outlier_vector[int(index)] = 1
        outlier_regressor = np.hstack((outlier_regressor, outlier_vector))

    outlier_regressor = outlier_regressor[:,1::]

    # load motion regressors
    motion_regressor=np.genfromtxt(motreg_file)

    # extract high variance confounds in wm/csf masks from motion corrected data
    csf_regressor=high_variance_confounds(time_nii, mask_img=csf_nii, detrend=True)
    #global_regressor=high_variance_confounds(time_nii, mask_img=brain_nii, detrend=True)

    # create Nifti Masker for denoising
    denoiser=NiftiMasker(mask_img=brain_mask, standardize=True, detrend=True, high_pass=bandpass[1], low_pass=bandpass[0], t_r=tr)

    # denoise and return denoise data to img
    confounds=np.hstack((outlier_regressor, csf_regressor, motion_regressor, artefact_ts))
    denoised_data=denoiser.fit_transform(in_file, confounds=confounds)
    denoised_img=denoiser.inverse_transform(denoised_data)

    # save
    _, base, _ = split_filename(in_file)
    img_fname = base + '_denoised.nii.gz'
    nb.save(denoised_img, img_fname)

    data_fname = base + '_denoised.npy'
    np.save(data_fname, denoised_data)

    confound_fname = os.path.join(os.getcwd(), "all_confounds.txt")
    np.savetxt(confound_fname, confounds, fmt="%.10f")

    return os.path.abspath(img_fname), os.path.abspath(data_fname), confound_fname


def motion_regressors(motion_params, order=0, derivatives=1):
    """Compute motion regressors upto given order and derivative
    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    from nipype.utils.filemanip import filename_to_list
    import numpy as np
    import os

    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        out_params = params
        for d in range(1, derivatives + 1):
            cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                 params))
            out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
        out_params2 = out_params
        for i in range(2, order + 1):
            out_params2 = np.hstack((out_params2, np.power(out_params, i)))
        filename = os.path.join(os.getcwd(), "motion_regressor_der%d_ord%d.txt" % (derivatives, order))
        np.savetxt(filename, out_params2, fmt="%.10f")
        out_files.append(filename)
    return out_files

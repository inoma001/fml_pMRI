#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:40:39 2018

@author: mike germuska
"""
code_ver="1.0.0"

import helper_functions as hf
import ML_functions as mf
import nibabel as nib


# get command line input parameters
args=hf.parse_cmdln()
# process command line parameters, put in dict and save to csv p_file
print('importing data')
images_dict,d_phys,d_analysis,d_scan_par = hf.process_cmdln(args,code_ver)
# calculate CMRO2, CBF0, and OEF0 using ANN
d_analysis['M0_cut']=8000  # value depends on image bit-depth and receiver gain
CMRO20,CBF0,OEF0, OEF_se = mf.calc_cmro2(images_dict,d_phys,d_scan_par,d_analysis)

print('saving data')
empty_header=nib.Nifti1Header()

OEF_img=nib.Nifti1Image(OEF0, images_dict['echo1_img'].affine, empty_header)
nib.save(OEF_img, d_analysis['outpath'] + 'OEF0.nii.gz')

OEF_se_img=nib.Nifti1Image(OEF_se, images_dict['echo1_img'].affine, empty_header)
nib.save(OEF_se_img, d_analysis['outpath'] + 'OEF0_se.nii.gz')

CBF_img=nib.Nifti1Image(CBF0, images_dict['echo1_img'].affine, empty_header)
nib.save(CBF_img, d_analysis['outpath'] + 'CBF0.nii.gz')

CMRO2_img=nib.Nifti1Image(CMRO20, images_dict['echo1_img'].affine, empty_header)
nib.save(CMRO2_img, d_analysis['outpath'] + 'CMRO20.nii.gz')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:41:04 2018

@author: mike germuska
"""
import argparse
import csv
import os
import errno
import nibabel as nib


def calc_SaO2(PaO2):
    SaO2=1/((23400/(PaO2**3+150*PaO2))+1)
    return(SaO2)
    
def calc_CaO2(PaO2,Hb):
    fi=1.34;
    eps=0.000031;
    SaO2=calc_SaO2(PaO2)
    CaO2=fi*Hb*SaO2+PaO2*eps; 
    return(CaO2)
    

def parse_cmdln():
    parser=argparse.ArgumentParser()

    parser.add_argument("-p","--p_file", help="csv parameter file that contains all info for analysis, must include the following keys: Hb, PaO20, dPaO2, PLD, echo1_fn, echo2_fn, M0_fn")
    parser.add_argument("-out","--out_path", help="Output pathname for saving the results")
    
#    Required if not using a p_file. if p_file specifed these paramters will overwrite the values from the file
    parser.add_argument("-e1","--echo1", help="echo 1 filename (Required if p_file not used)")
    parser.add_argument("-e2","--echo2", help="echo 2 filename (Required if p_file not used)")
    parser.add_argument("-M0", help="M0 filename (Required if p_file not used)")

    # optional arguments to overwrite defaults or contents of p_file
    parser.add_argument("--PaO20", help="basline PaO2 value in mmHg",type=float)
    parser.add_argument("--dPaO2", help="change in PaO2 due to hyperoxia challenge mmHg",type=float)
    parser.add_argument("--Hb", help="Hb (units)",type=float)
    parser.add_argument("--PLD", help="PLD in seconds",type=float)
 
    parser.parse_args()
    args=parser.parse_args()
    
#    add file extension to p_file if needed
    if args.p_file:
        if args.p_file[-4:]!='.csv':
            args.p_file=args.p_file + '.csv'
    
    # check that input filenames have been specified if not using a p_file
    if args.p_file==None:
        try:
           if args.echo1==None:
               raise Exception("Input filenames must be specified if not using a p_file")
           if args.echo2==None:
               raise Exception("Input filenames must be specified if not using a p_file")
           if args.M0==None:
               raise Exception("Input filenames must be specified if not using a p_file")
        except Exception as error:
            print(error)
            raise SystemExit(0)
    return args
        
def process_cmdln(args,code_ver):
    
    # list of essential dictionary keys that need to be in p_file if used
    key_list=['Hb','PaO20','dPaO2','PLD','echo1_fn','echo2_fn', 'M0_fn']
    
    # create dictionaries with default values 
    d_analysis={}
    d_analysis['code_ver']=code_ver # save code version with results
    # set default path for results as a new results dir in pwd
    d_analysis['outpath']=os.getcwd() + '/results/'
        
    d_phys={}
    d_phys['Hb']=0.15
    d_phys['PaO20']=120
    d_phys['dPaO2']=220
    d_phys['k']=3
    
    d_scan_par={}
    d_scan_par['PLD']=1.5
    d_scan_par['slice_delay']=0.0367
    # tag fixed at 1.5 during ANN training!!
    
    if args.p_file==None:
        d_analysis['echo1_fn']=args.echo1
        d_analysis['echo2_fn']=args.echo2
        d_analysis['M0_fn']=args.M0   
    else: # read data from csv file and to dictionaries and convert appropriate data to floats... no error checking of csv file contents
        read_dict={}
        with open(args.p_file, mode='r') as f:
            r = csv.reader(f)
            for rows in r:
                read_dict[rows[0]]=rows[1]
# check all essential keys are in p_file
        if len(set(key_list) - read_dict.keys()) > 0:
            print('please check the selected p_file contains entries for: ', set(key_list) - read_dict.keys())
            raise SystemExit(0)
        for key in read_dict:
            if key in d_phys:
                d_phys[key]=float(read_dict[key])
            if key in d_scan_par:
                d_scan_par[key]=float(read_dict[key])
            if key in d_analysis:
                d_analysis[key]=read_dict[key]
 #    read in keys for input files (which should not in d_analysis dictionary yet)
            d_analysis['echo1_fn']=read_dict['echo1_fn']
            d_analysis['echo2_fn']=read_dict['echo2_fn'] 
            d_analysis['M0_fn']=read_dict['M0_fn'] 
            
 # overwrite parametes if optional arguments specified
    if args.PaO20:
        d_phys['PaO20']=args.PaO20
    if args.dPaO2:    
        d_phys['dPaO2']=args.dPaO2
    if args.Hb:
        d_phys['Hb']=args.Hb
    if args.PLD:
        d_scan_par['PLD']=args.PLD
      
#         add calulated values - these need to be re-calculated for consistency
    d_phys['CaO20']=round(calc_CaO2(d_phys['PaO20'],d_phys['Hb']),3)
    d_phys['SaO20']=round(calc_SaO2(d_phys['PaO20']),3)
        
               
#        change output path if specified
    if args.out_path:
#        add '/' at end of out_path if does not exist
        if args.out_path[-1]!='/':
           args.out_path=args.out_path + '/'
#        if outpath starts with a '/' assume it is absolute path and use as is
        if args.out_path[0]=='/':
            d_analysis['outpath']=args.out_path 
        else: # if we don't think it is an absolute path add pwd to correct this
            d_analysis['outpath']=os.getcwd() + '/' + args.out_path
    
    if args.echo1:
        d_analysis['echo1_fn']=args.echo1
    if args.echo2:
        d_analysis['echo2_fn']=args.echo2
    if args.M0:
        d_analysis['M0_fn']=args.M0

# Check if input files exist .... and load
    try:
        echo1_img=nib.load(d_analysis['echo1_fn'])
        echo1_data=echo1_img.get_data()
    except Exception as error:
            print(error)
            raise SystemExit(0)
    try:
        echo2_img=nib.load(d_analysis['echo2_fn'])
        echo2_data=echo2_img.get_data()
    except Exception as error:
            print(error)
            raise SystemExit(0)
    try:
        M0_img=nib.load(d_analysis['M0_fn'])
        M0_data=M0_img.get_data()
        
#        this might need adjusting for M0 data but probably not as wil pre-process AP and PA data
#        to one 3D dataset.
    except Exception as error:
            print(error)
            raise SystemExit(0)      
        
    images_dict={}
    images_dict['echo1_img']=echo1_img; # can be useful for creating NiftiImage for saving data
    images_dict['echo1_data']=echo1_data
    images_dict['echo2_img']=echo2_img 
    images_dict['echo2_data']=echo2_data     
    images_dict['M0_img']=M0_img
    
    import numpy as np
    M0_shape=np.shape(M0_data)

    if np.size(M0_shape) >3:
        images_dict['M0_data']=M0_data[:,:,:,0] 
    else:
        images_dict['M0_data']=M0_data
    
# Save p_file to output directory for record of analysis / easy re-run
    
    try:
        os.makedirs(d_analysis['outpath'])
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory for results already exists. Please specify an alternative output path')
            raise SystemExit(0)
        else:
            raise
    
    with open(d_analysis['outpath'] + 'p_file.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(d_phys.items())
        w.writerows(d_scan_par.items())
        w.writerows(d_analysis.items())
    
        
    return images_dict,d_phys,d_analysis,d_scan_par

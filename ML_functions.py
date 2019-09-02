
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:47:07 2018

@author: mike germuska
"""
import numpy as np
import pickle
import pyfftw
from scipy import linalg
from scipy import sparse


def create_HP_filt(flength,cutoff,TR):
    cut=cutoff/TR
    sigN2=(cut/np.sqrt(2))**2
    K=linalg.toeplitz(1/np.sqrt(2*np.pi*sigN2)*np.exp(-np.linspace(0,flength,flength)**2/(2*sigN2)))
    K=sparse.spdiags(1/np.sum(K,axis=0),0,flength,flength)*K
    H=np.zeros([flength,flength])
    X=np.array([np.ones(flength), range(1,flength+1)])
    X=np.transpose(X)
    for i in range(flength):
        W=np.diag(K[i])
        Hat=np.dot(np.dot(X,linalg.pinv(np.dot(W,X))),W)
        H[i]=Hat[i]
    HPfilt=np.eye(flength)-H
    return HPfilt


def calc_cmro2(images_dict,d_phys,d_scan_par,d_analysis):

    print('pre-processing ASL and BOLD data')

    #scale echo1 1 data by M0 and threshold out low M0 values (also scale by 100)
    x_axis,y_axis,no_slices,datapoints=np.shape(images_dict['echo1_data'])
    image_data=np.zeros([x_axis,y_axis,no_slices,datapoints])
    for i in range(datapoints):
        with np.errstate(divide='ignore',invalid='ignore'):
            image_data[:,:,:,i]=100*(np.divide(images_dict['echo1_data'][:,:,:,i],images_dict['M0_data']))
            image_data[:,:,:,i][images_dict['M0_data']<d_analysis['M0_cut']]=0
    
    flow_data=pyfftw.empty_aligned([x_axis,y_axis,no_slices,datapoints-2]) # pre-allocate n-byte aligned empty array
    # matrix surround subtraction for both c-(t0+t2)/2  and t+(c0+c2) to get perfusion data
    # for even data points
    flow_data=image_data[:,:,:,1:-1]-(image_data[:,:,:,0:-2]+image_data[:,:,:,2:])/2
    # for od data points
    flow_odd=-image_data[:,:,:,1:-1]+(image_data[:,:,:,0:-2]+image_data[:,:,:,2:])/2
    # add in odd data points
    flow_data[:,:,:,1::2]=flow_odd[:,:,:,1::2]    

    # surround average to get BOLD data
    bold_data=(images_dict['echo2_data'][:,:,:,1:-1]+(images_dict['echo2_data'][:,:,:,0:-2]+images_dict['echo2_data'][:,:,:,2:])/2)/2
    # convert into percent signal change
    per_bold=pyfftw.empty_aligned([x_axis,y_axis,no_slices,datapoints-2]) # pre-allocate n-byte aligned empty array

    cut=320; 
    HPfilt=create_HP_filt(247,cut,4.4)

    baseline=np.mean(bold_data[:,:,:,0:20],axis=3)
    for i in range(datapoints-2):
        with np.errstate(divide='ignore', invalid='ignore'):
            per_bold[:,:,:,i]=np.divide(bold_data[:,:,:,i],baseline)
            per_bold[:,:,:,i][baseline==0]=0
    per_bold=(per_bold-1)
        
    #     HP filter data
    print('HP filt BOLD data')
    for i in range(x_axis):    
        for j in range(y_axis): 
            for k in range(no_slices): 
                per_bold[i,j,k,:]=per_bold[i,j,k,:]-np.mean(per_bold[i,j,k,0:20])
                per_bold[i,j,k,:]=np.dot(HPfilt,per_bold[i,j,k,:])
                per_bold[i,j,k,:]=per_bold[i,j,k,:]-np.mean(per_bold[i,j,k,0:20])
    
    
    print('pyfftw FFT')
    
        
#    
    # calculate the FFT of BOLD and ASL data
    # FFTW is faster than numpy fft so use this.

#    import pre-computed fftw wisdom for these datasets for significant speed-up
    fft_wisdom=pickle.load(open('fft_wisdom.sav', 'rb')) 
    pyfftw.import_wisdom(fft_wisdom)
    pyfftw.interfaces.cache.enable()
    
    BOLD_fft=pyfftw.interfaces.numpy_fft.fft(per_bold)
    ASL_fft=pyfftw.interfaces.numpy_fft.fft(flow_data)
    
#    export and save fftw wisdom to speed up analysis
#    fft_wisdom=pyfftw.export_wisdom()
#    pickle.dump(fft_wisdom,open('fft_wisdom.sav', 'wb'))
   
    # Now calculate CBF0
    print('predicting CBF0')
    PLD_vect=np.linspace(d_scan_par['PLD'],d_scan_par['PLD']+no_slices*d_scan_par['slice_delay'], num=no_slices)
    PLD_mat=np.tile(PLD_vect, (x_axis,y_axis,1))


    array_elements=15;
    ASL_ML_array=np.empty([x_axis,y_axis,no_slices,5+4*array_elements])
    ASL_ML_array[:,:,:,0]=d_phys['Hb']
    ASL_ML_array[:,:,:,1]=d_phys['dPaO2']
    ASL_ML_array[:,:,:,2]=d_phys['SaO20']
    ASL_ML_array[:,:,:,3]=d_phys['CaO20']
    ASL_ML_array[:,:,:,4]=PLD_mat
    ASL_ML_array[:,:,:,5:5+array_elements]=np.absolute(ASL_fft[:,:,:,0:array_elements])
    ASL_ML_array[:,:,:,5+array_elements:5+2*array_elements]=np.angle(ASL_fft[:,:,:,0:array_elements])
    ASL_ML_array[:,:,:,5+2*array_elements:5+3*array_elements]=np.absolute(BOLD_fft[:,:,:,0:array_elements])
    ASL_ML_array[:,:,:,5+3*array_elements:5+4*array_elements]=np.angle(BOLD_fft[:,:,:,0:array_elements])

    
    filename='fft_CBF0_50k_xtree_001_model.sav'
    net=pickle.load(open(filename, 'rb'))    
    filename='fft_CBF0_50k_xtree_001_scaler.sav'
    scaler=pickle.load(open(filename, 'rb'))  

    ASL_ML_array=np.reshape(ASL_ML_array,(x_axis*y_axis*no_slices, 5+4*array_elements))
    X_train_scaled=scaler.transform(ASL_ML_array)
   
    CBF0_vect=net.predict(X_train_scaled)*100
    CBF0=np.reshape(CBF0_vect, (x_axis,y_axis,no_slices))
      
    
    CBF0[CBF0<0]=0
    CBF0[CBF0>250]=250
    CBF0[images_dict['M0_data']<d_analysis['M0_cut']]=0

    
    print('predicting CMRO2')    
    array_elements=15; 
    
    ML_array=np.zeros([x_axis,y_axis,no_slices,6+array_elements*4])
    ML_array[:,:,:,0]=d_phys['Hb']
    ML_array[:,:,:,1]=d_phys['dPaO2']
    ML_array[:,:,:,2]=d_phys['SaO20']
    ML_array[:,:,:,3]=d_phys['CaO20']
    ML_array[:,:,:,4]=PLD_mat
    ML_array[:,:,:,5]=CBF0
    ML_array[:,:,:,6:6+array_elements]=np.absolute(ASL_fft[:,:,:,0:array_elements])
    ML_array[:,:,:,6+array_elements:6+2*array_elements]=np.angle(ASL_fft[:,:,:,0:array_elements])
    ML_array[:,:,:,6+2*array_elements:6+3*array_elements]=np.absolute(BOLD_fft[:,:,:,0:array_elements])
    ML_array[:,:,:,6+3*array_elements:6+4*array_elements]=np.angle(BOLD_fft[:,:,:,0:array_elements])
#reshape ML array to 2D for use with the scaler and ANN
    ML_array=np.reshape(ML_array,(x_axis*y_axis*no_slices, 6+4*array_elements))

#    ensemble fitting
    import os
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    ensembleDirectory = os.path.join(scriptDirectory, 'CBF_OEF_ensemble_sav/')	
    file_list = os.listdir(ensembleDirectory)
   
    CMRO2_array=np.zeros([x_axis,y_axis,no_slices,int(len(file_list)/2)])
    OEF_array=np.zeros([x_axis,y_axis,no_slices,int(len(file_list)/2)])
    
    array_counter=-1
    for i in range(len(file_list)):
        current_file=file_list[i]
        if current_file[-9:-4]=='model':
            filename='CBF_OEF_ensemble_sav/' + current_file
            
            print(filename)         
            net=pickle.load(open(filename, 'rb'))
            filename='CBF_OEF_ensemble_sav/' + current_file[0:-9] + 'scaler.sav'
            scaler=pickle.load(open(filename, 'rb')) 
            X_train_scaled=scaler.transform(ML_array)
            CBF_OEF_vect=net.predict(X_train_scaled)*100
            CBF_OEF_local=np.reshape(CBF_OEF_vect, (x_axis,y_axis,no_slices))

            array_counter+=1
            print(array_counter)
            CMRO2_array[:,:,:,array_counter]=CBF_OEF_local*39.34*d_phys['CaO20']
 


#    convert whole CMRO2 array to OEF array. limit values to 0-1. 
          
    with np.errstate(divide='ignore', invalid='ignore'):
        OEF_array= np.divide( CMRO2_array , (39.34*d_phys['CaO20']*CBF0)[:,:,:,None])            
    
    OEF_array[np.isnan(OEF_array)]=0
    OEF_array[np.isinf(OEF_array)]=0   
    OEF=np.mean(OEF_array,3) 
    
    
#   remove impossible answers 
    OEF[OEF>=1]=0
    OEF[OEF<0]=0
         
    OEF_se=np.std(OEF_array,3,ddof=1) 
    OEF_se[images_dict['M0_data']<d_analysis['M0_cut']]=0
    
#     recalculate CMRO2 
    OEF[images_dict['M0_data']<d_analysis['M0_cut']]=0
    CMRO2=OEF*39.34*d_phys['CaO20']*CBF0
               
    return CMRO2, CBF0, OEF, OEF_se


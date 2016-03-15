#!/usr/bin/env python
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.freesurfer as fs
from nipype.interfaces.dcm2nii import Dcm2nii
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nibabel as nb
import numpy as np
import argparse
import textwrap
import re, glob
import os                                    # system functions
import sys
import shutil
from nipype.workflows.dmri.fsl.dti import create_bedpostx_pipeline
from nipype.workflows.dmri.fsl.dti import bedpostx_parallel


def dti_preprocessing(args):
    '''
    Write it later
    '''
    dataLoc = os.getcwd()#'/Volumes/CCNC_3T/KMA'
    subject_list = args.subjects

    # Make exclusion mask in order to exclude tracks
    # going towards posterior paths from the thalamus
    for subject in subject_list:
        dtiDir = os.path.join(dataLoc, subject, 'DTI')

        # Check dicom
        dicoms = glob.glob(dtiDir+'/*dcm') + \
                glob.glob(dtiDir+'/*IMA') + \
                glob.glob(dtiDir+'/dicom/*dcm') + \
                glob.glob(dtiDir+'/dicom/*IMA')

        if dicoms:
            dicomDir = os.path.join(dtiDir, 'dicom')
            if not os.path.isdir(dicomDir):
                os.mkdir(dicomDir)

            try:
                for dicomFile in dicoms:
                    shutil.move(dicomFile, dicomDir)
            except:
                pass
            dicoms = glob.glob(dicomDir+'/*dcm') + glob.glob(dicomDir+'/*IMA')

            nifti = glob.glob(dtiDir+'/*nii.gz')
            # if no nifti
            if not nifti:
                converter = Dcm2nii(
                        source_names = dicoms,
                        gzip_output = True,
                        output_dir = dtiDir)
                converter.run()

        # raw data
        data = glob.glob(dtiDir+'/2*DTI*.nii.gz')[0]
        bvec = glob.glob(dtiDir+'/2*.bvec')[0]
        bval = glob.glob(dtiDir+'/2*.bval')[0]

        # preprocessed data
        eddy_out = os.path.join(dtiDir, 'data.nii.gz')
        newBvec = os.path.join(dtiDir, 'bvecs')
        newBval = os.path.join(dtiDir, 'bvals') # just a copy
        nodif = os.path.join(dtiDir, 'nodif.nii.gz')
        nodif_brain = os.path.join(dtiDir, 'nodif_brain.nii.gz')
        nodif_brain_mask = os.path.join(dtiDir, 'nodif_brain_mask.nii.gz')
        fa_map = os.path.join(dtiDir, 'dti_FA.nii.gz')
        bedpostxdir = os.path.join(subject, 'DTI.bedpostX')


        if not os.path.isfile(newBval):
            shutil.copy(bval, newBval)

        if not os.path.isfile(eddy_out):
            eddy = fsl.EddyCorrect(
                    in_file = data,
                    out_file = eddy_out)
            eddy.run()

        if not os.path.isfile(newBvec):
            bvecCorrectCommand = 'bash /Volumes/CCNC_3T/KMA/kmaproject/fdt_rotate_bvecs.sh \
                    {origBvec} {newBvec} {ecclog}'.format(
                    origBvec = bvec,
                    newBvec = newBvec,
                    ecclog = eddy_out.split('nii.gz')[0]+'ecclog'
                    )
            print bvecCorrectCommand
            os.popen(bvecCorrectCommand).read()


        if not os.path.isfile(nodif):
            extractROI = fsl.ExtractROI(
                            in_file = eddy_out,
                            t_min = 0,
                            t_size = 1,
                            roi_file = nodif,)
            extractROI.run()

        if not os.path.isfile(nodif_brain_mask):
            bet = fsl.BET(
                    in_file = eddy_out,
                    frac = .35,
                    mask = True,
                    out_file = nodif_brain
                    )
            bet.run()
                    

        if not os.path.isfile(fa_map):
            dtifit = fsl.DTIFit(
                    bvals = bval,
                    bvecs = newBvec,
                    dwi = eddy_out,
                    mask = nodif_brain_mask,
                    base_name = dtiDir+'/DTI')
            dtifit.run()


        params = dict(n_fibres = 2,
                fudge = 1,
                burn_in = 1000,
                n_jumps = 1250,
                sample_every = 25)

        bpwf = create_bedpostx_pipeline('nipype_bedpostx', params)
        bpwf.inputs.inputnode.dwi = eddy_out
        bpwf.inputs.inputnode.mask = nodif_brain_mask
        bpwf.inputs.inputnode.bvecs = newBvec
        bpwf.inputs.inputnode.bvals = newBval
        bpwf.run()


        #params = dict(n_fibres = 2, fudge = 1, burn_in = 1000,
                #n_jumps = 1250, sample_every = 25)
        #bpwf = bedpostx_parallel('nipype_bedpostx_parallel', params)
        #bpwf.inputs.inputnode.dwi = eddy_out
        #bpwf.inputs.inputnode.mask = nodif_brain_mask
        #bpwf.inputs.inputnode.bvecs = newBvec
        #bpwf.inputs.inputnode.bvals = bval
        #bpwf.run(plugin='CondorDAGMan')

                #if not os.path.isdir(bedpostxdir):
        #bedpostx = fsl.BEDPOSTX5(
                #bpx_directory = bedpostxdir,
                ##logdir = bedpostxdir,
                #out_dir = bedpostxdir,
                #bvecs = newBvec,
                #bvals = bval,
                #dwi = eddy_out,
                #mask = nodif_brain_mask,
                #uskmae_gpu=True,
                #n_fibres = 2)
        #bedpostx.run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            {codeName} : Runs thalamo-cortical probabilistic tractography 
            ========================================
            '''.format(codeName=os.path.basename(__file__))))
    parser.add_argument(
        '-s', '--subjects',
        help='subject list',
        nargs='+')
    #parser.add_argument(
        #'-side', '--side',
        #help='side')
    args = parser.parse_args()

    dti_preprocessing(args)

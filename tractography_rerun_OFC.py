#!/usr/bin/env python
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nibabel
import numpy as np
import argparse
import textwrap
import os                                    # system functions

def tractography(args):
    '''
    It runs probabilistic tractography
    from thalamus to orbitofrontal cortex.
    It requires FSL, Freesurfer and nipype.
    All cortical ROIs in the contralateral hemisphere 
    and the cerebellum are used as exclusion mask.
    '''
    dataLoc = '/Volumes/CCNC_3T_2/kcho/ccnc/GHR_project'
    subject_list = args.subjects

    '''
    Make exclusion mask in order to exclude tracks
    going towards posterior paths from the thalamus
    '''
    for subject in subject_list:
        for side in ['lh', 'rh']:
            roiLoc = os.path.join(dataLoc, subject_list, 'ROI')
            thalamusROI = os.path.join(roiLoc, side+'_thalamus.nii.gz')
            newROI = os.path.join(roiLoc,side+'_post_thal_excl_mask.nii.gz')
                    
            if not os.path.isfile(newROI):
                thalamusPosterior(thalamusROI)


    '''Dictionary for datasource'''
    info = dict(dwi=[['subject_id', 'data']],
                bvecs=[['subject_id', 'bvecs']],
                bvals=[['subject_id', 'bvals']],
                seed_file=[['subject_id', 'thalamus']],
                target_mask=[['subject_id', 'OFC']],
                exclusion_masks=[['subject_id', 
                    ['LPFC', 'LTC', 'MPFC','MTC','OCC','OFC','PC','SMC']]],
                thsample = [['subject_id',
                    ['merged_th1samples','merged_th2samples']]],
                phsample = [['subject_id',
                    ['merged_ph1samples','merged_ph2samples']]],
                fsample = [['subject_id',
                    ['merged_f1samples','merged_f2samples']]],
                matrix = [['subject_id','FREESURFERT1toNodif.mat']],
                bet_mask = [['subject_id','nodif_brain_mask']],
                fsLoc = [['subject_id','freesurfer']],
                aseg = [['subject_id','aseg']],
                aparc_aseg= [['subject_id','aparc+aseg']],
                )

    '''Subject name list setting into identity interface'''
    infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                         name="infosource")
    infosource.iterables = ('subject_id', subject_list)


    '''Data source setting'''
    datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                                   outfields=info.keys()),
                         name = 'datasource')
    datasource.inputs.template = "%s/%s"
    datasource.inputs.base_directory = os.path.abspath(dataLoc)
    datasource.inputs.field_template = dict(dwi='%s/dti/%s.nii.gz',
                                            bvecs='%s/dti/%s',
                                            bvals='%s/dti/%s',
                                            seed_file='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            target_mask='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            exclusion_masks='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            matrix='%s/registration/%s',
                                            bet_mask='%s/dti/%s.nii.gz',
                                            thsample='%s/DTI.bedpostX/%s.nii.gz',
                                            fsample='%s/DTI.bedpostX/%s.nii.gz',
                                            phsample='%s/DTI.bedpostX/%s.nii.gz',
                                            fsLoc='%s/%s',
                                            aseg='%s/FREESURFER/mri/%s.mgz',
                                            aparc_aseg='%s/FREESURFER/mri/%s.mgz',
                                            )
    datasource.inputs.template_args = info
    datasource.inputs.sort_filelist = True


    '''Freesurfer'''
    insulaExtract= pe.Node(interface=fs.Binarize(out_type='nii.gz'), name='insula')
    if args.side == 'lh':
        insulaExtract.inputs.match = [1035, 2035]
    else:
        insulaExtract.inputs.match = [1035, 2035]

    brainStemExtract = pe.Node(interface=fs.Binarize(out_type='nii.gz'), name='brainStem')
    brainStemExtract.inputs.match = [16, 6, 7, 8, 45, 46, 47]

    '''
    Merge output from freesurfer binarize : 
    15 because 8 + 7 cortical ROIs (infile is insula)
    '''
    cortexStopRoiMaker = pe.Node(interface=fsl.MultiImageMaths(), name='add_masks')
    cortexStopRoiMaker.inputs.op_string = '-add %s '* 15

    '''
    2   Left-Cerebral-White-Matter              245 245 245 0
    41  Right-Cerebral-White-Matter             0   225 0   0
    '''
    wmExtract = pe.Node(interface=fs.Binarize(out_type='nii.gz'), name='wmExtract')
    if args.side == 'lh':
        # needs to put the contralateral hemisphere
        wmExtract.inputs.match = [41]
    else:
        wmExtract.inputs.match = [2]


    



    '''Probabilistic tractography'''
    probtrackx = pe.Node(interface=fsl.ProbTrackX2(), name='probtrackx_OFC')
    probtrackx.inputs.onewaycondition= True
    #Below are the default options
    #probtrackx.inputs.waycond = 'AND'
    #probtrackx.inputs.c_thresh = 0.2
    #probtrackx.inputs.n_steps = 2000
    #probtrackx.inputs.step_length = 0.5
    #probtrackx.inputs.n_samples = 5000
    #probtrackx.inputs.opd = True
    #probtrackx.inputs.os2t = True
    #probtrackx.inputs.loop_check = True


    '''Data sink'''
    datasink = pe.Node(interface=nio.DataSink(),name='datasink')
    datasink.inputs.base_directory = os.path.abspath('/Volumes/CCNC_3T_2/kcho/ccnc/GHR_project/OFC_tractography_wm')
    datasink.inputs.substitutions = [('_variable', 'variable'),
                                     ('_subject_id_', 'subject_id')]

    ''' Workflow '''
    dwiproc = pe.Workflow(name="dwiproc_OFC")
    dwiproc.base_dir = os.path.abspath('tractography_OFC')
    dwiproc.connect([
                        (infosource,datasource,[('subject_id', 'subject_id')]),
                        #(datasource, insulaExtract, [('aparc_aseg', 'in_file')]),
                        (datasource, brainStemExtract, [('aseg', 'in_file')]),
                        (datasource, wmExtract, [('aseg', 'in_file')]),
                        (wmExtract, cortexStopRoiMaker,[('binary_file', 'in_file')]),
                        (datasource,cortexStopRoiMaker,[(('exclusion_masks',get_opposite), 'operand_files')]),
                        #(cortexStopRoiMaker, allStopRoiMaker,[('out_file', 'in_file')]),
                        #(brainStemExtract, allStopRoiMaker,[('out_file', 'operand_files')]),
                        #(allStopRoiMaker, probtrackx,[('out_file', 'avoid_mp')]),
                        (cortexStopRoiMaker, probtrackx,[('out_file', 'avoid_mp')]),
                        (datasource,probtrackx,[('seed_file','seed'),
                                                   ('target_mask','stop_mask'),
                                                   ('target_mask','waypoints'),
                                                   ('bet_mask','mask'),
                                                   ('phsample','phsamples'),
                                                   ('fsample','fsamples'),
                                                   ('thsample','thsamples'),
                                                   ('matrix','xfm'),
                                                   ]),
                        (probtrackx,datasink,[('fdt_paths','probtrackx.@fdt_paths'),
                            ('log', 'probtrackx.@log'),
                            ('particle_files', 'probtrackx.@particle_files'),
                            ('targets', 'probtrackx.@targets'),
                            ('way_total', 'probtrackx.@way_total'),
                            ])
                    ])

    '''Parallel processing'''
    dwiproc.run(plugin='MultiProc', plugin_args={'n_procs' : 8})


def get_opposite(roiList):
    import os
    if os.path.basename(roiList[0]).startswith('lh'):
        newList = [x.replace('lh','rh') for x in roiList]
        for i in roiList:
            if not os.path.basename(i).startswith('lh_OFC'):
                newList.append(i)
        print [os.path.basename(x) for x in newList]

    else:
        newList = [x.replace('rh','lh') for x in roiList]
        for i in roiList:
            if not os.path.basename(i).startswith('rh_OFC'):
                newList.append(i)
        print newList
    return newList




def thalamusPosterior(thalamusImg):
    roiLoc = os.path.dirname(thalamusImg)
    side = os.path.basename(thalamusImg)[:2]


    # ROI load
    f = nb.load(thalamusImg)
    data = f.get_data()

    # find lowest z coordinate
    z_length = data.shape[2]
    for sliceNum in range(z_length):
        if 1 in data[:,:,sliceNum]:
            break

    newArray = np.zeros_like(data)
    newArray[:,:,sliceNum] = 1

    newROI = os.path.join(roiLoc,side+'_post_thal_excl_mask.nii.gz')
    nb.Nifti1Image(newArray, f.affine).to_filename(newROI)



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
    parser.add_argument(
        '-side', '--side',
        help='side')
    args = parser.parse_args()

    tractography(args)

#!/usr/bin/env python
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nibabel as nb
import numpy as np
import argparse
import textwrap
import os                                    # system functions

def tractography(args):
    '''
    It runs probabilistic tractography
    from thalamus to orbitofrontal cortex.
    It requires FSL, Freesurfer and nipype.

    Exclusion masks are
    - All cortical ROIs in the contralateral hemisphere 
    - Brain stem
    - Cerebellum 
    - Ipsilateral temporal lobe
    '''
    dataLoc = '/Volumes/CCNC_3T_2/kcho/ccnc/GHR_project'
    subject_list = args.subjects
    atlasLoc = '/usr/local/fsl/data/atlases/MNI/MNI-maxprob-thr0-1mm.nii.gz'

    # Make exclusion mask in order to exclude tracks
    # going towards posterior paths from the thalamus
    for subject in subject_list:
        for side in ['lh', 'rh']:
            roiLoc = os.path.join(dataLoc, subject, 'ROI')
            thalamusROI = os.path.join(roiLoc, side+'_thalamus.nii.gz')
            plan_exclusion_mask = os.path.join(roiLoc,side+'_post_thal_TC_excl_mask.nii.gz')
                    
            regLoc = os.path.join(dataLoc, subject, 'registration')
            temporalExROI = 'MNI_temporal_mask.nii.gz'
            temporalExROI_sub = os.path.join(roiLoc, 'temporalExROI.nii.gz')
            mni2subj = os.path.join(roiLoc, 'mni2subj.mat')
            subjBrain = os.path.join(dataLoc, subject, 'FREESURFER',
                    'mri','brain.nii.gz')

            if not os.path.isfile(temporalExROI_sub):
                # Registration
                if not os.path.isfile(mni2subj):
                    MNIreg = fsl.FLIRT(
                            in_file = atlasLoc,
                            reference = subjBrain,
                            interp = 'nearestneighbour',
                            out_matrix_file = mni2subj)
                    MNIreg.run()

                # ROI extraction
                if not os.path.isfile(temporalExROI):
                    extractTC = fsl.ImageMaths(
                            in_file = atlasLoc,
                            op_string = '-thr 8 -uthr 8',
                            out_file = temporalExROI,
                            )
                    extractTC.run()

                # Apply registration
                if not os.path. isfile(temporalExROI_sub):
                    TC_to_subj = fsl.ApplyXfm(
                            in_file = temporalExROI,
                            reference = subjBrain,
                            interp = 'nearestneighbour',
                            in_matrix_file = mni2subj,
                            out_file = temporalExROI_sub)
                    TC_to_subj.run()

            if not os.path.isfile(plan_exclusion_mask):
                thal_TC_posterior(thalamusROI, temporalExROI_sub)


    # Dictionary for datasource
    info = dict(dwi=[['subject_id', 'data']],
                bvecs=[['subject_id', 'bvecs']],
                bvals=[['subject_id', 'bvals']],
                seed_file=[['subject_id', 'thalamus']],
                target_mask=[['subject_id', 'OFC']],
                exclusion_masks=[['subject_id', 
                    ['LPFC', 'LTC', 'MPFC','MTC',
                     'OCC','OFC','PC','SMC']]],
                posterior_em = [['subject_id', 'post_thal_TC_excl_mask']],
                TC_em = [['subject_id', 'temporalExROI']],
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

    # Subject name list setting into identity interface
    infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                         name="infosource")
    infosource.iterables = ('subject_id', subject_list)


    # Data source setting
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
                                            posterior_em='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            TC_em='%s/ROI/%s.nii.gz',
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

    brainStemExtract = pe.Node(interface=fs.Binarize(out_type='nii.gz'), name='brainStem')
    brainStemExtract.inputs.match = [16, 6, 7, 8, 45, 46, 47]

    wmExtract = pe.Node(interface=fs.Binarize(out_type='nii.gz'), name='wmExtract')
    #    2   Left-Cerebral-White-Matter              245 245 245 0
    #    41  Right-Cerebral-White-Matter             0   225 0   0
    if args.side == 'lh':
        # needs to put the contralateral hemisphere
        wmExtract.inputs.match = [41]
    else:
        wmExtract.inputs.match = [2]

    
    # Merge exclusion ROIs
    # - brain stem
    # - contralateral hemisphere white matter 
    # - slice posterior to the thalamus 
    # - temporal lobe
    bs_add_wm = pe.Node(interface=fsl.MultiImageMaths(), name='add_wm')
    bs_add_wm.inputs.op_string = '-add %s'

    bs_add_wm_add_ex = pe.Node(interface=fsl.MultiImageMaths(), name='add_posterior_plane')
    bs_add_wm_add_ex.inputs.op_string = '-add %s'

    bs_add_wm_add_ex_add_tc = pe.Node(interface=fsl.MultiImageMaths(), name='add_temporal_lobe')
    bs_add_wm_add_ex_add_tc.inputs.op_string = '-add %s'
    # Probabilistic tractography
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


    # Data sink
    datasink = pe.Node(interface=nio.DataSink(),name='datasink')
    datasink.inputs.base_directory = os.path.abspath('/Volumes/CCNC_3T_2/kcho/ccnc/GHR_project/OFC_tractography_wm')
    datasink.inputs.substitutions = [('_variable', 'variable'),
                                     ('_subject_id_', 'subject_id')]

    # Workflow 
    dwiproc = pe.Workflow(name="dwiproc_OFC")
    dwiproc.base_dir = os.path.abspath('tractography_OFC_wm_pthalex')
    dwiproc.connect([
                        (infosource,datasource,[('subject_id', 'subject_id')]),
                        (datasource, brainStemExtract, [('aseg', 'in_file')]),
                        (datasource, wmExtract, [('aseg', 'in_file')]),
                        (brainStemExtract, bs_add_wm,[('binary_file', 'in_file')]),
                        (wmExtract, bs_add_wm,[('binary_file', 'operand_files')]),
                        (datasource, bs_add_wm_add_ex, [('posterior_em', 'operand_files')]),
                        (bs_add_wm, bs_add_wm_add_ex, [('out_file', 'in_file')]),
                        (datasource, bs_add_wm_add_ex_add_tc, [('TC_em', 'operand_files')]),
                        (bs_add_wm_add_ex, bs_add_wm_add_ex_add_tc, [('out_file', 'in_file')]),
                        (bs_add_wm_add_ex_add_tc, probtrackx,[('out_file', 'avoid_mp')]),
                        (bs_add_wm_add_ex_add_tc, datasink,[('out_file','exclusion_mask')]),
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

    # Parallel processing
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

def thal_TC_posterior(thalamusImg, TC_img):
    roiLoc = os.path.dirname(thalamusImg)
    side = os.path.basename(thalamusImg)[:2]


    # ROI load
    f_thal = nb.load(thalamusImg)
    data_thal = f_thal.get_data()

    f_TC = nb.load(TCamusImg)
    data_TC = f_TC.get_data()

    # find lowest z coordinate
    z_length = data_thal.shape[2]
    for sliceNumThal in range(z_length):
        if 1 in data_thal[:,:,sliceNumThal]:
            break

    z_length = data_thal.shape[2]
    for sliceNumTC in range(z_length):
        if 1 in data_TC[:,:,sliceNumTC]:
            break

    # lower the slice number the posterior the plane
    if sliceNumThal > sliceNumTC:
        sliceNum = sliceNumTC
    else:
        sliceNum = sliceNumThal


    newArray = np.zeros_like(data_thal)
    newArray[:,:,sliceNum] = 1

    plan_exclusion_mask = os.path.join(roiLoc,side+'_post_thal_TC_excl_mask.nii.gz')
    nb.Nifti1Image(newArray, f_thal.affine).to_filename(plan_exclusion_mask)



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

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
    from temporal cortex to frontal cortex.
    It requires FSL, Freesurfer and nipype.

    Temporal cortex labels are
    - Superior, Middle, and Inferior Temporal
    - Banks of the Superior Temporal Sulcus
    - Fusiform
    - Transverse Temporal
    - Entorhinal
    - Temporal Pole
    - Parahippocampal

    Frontal cortex labels are
    - Superior Frontal
    - Rostral and Caudal Middle Frontal
    - Pars Opercularis, Pars Triangularis, and Pars Orbitalis
    - Lateral and Medial Orbitofrontal
    - Precentral
    - Paracentral
    - Frontal Pole

    Exclusion masks are
    - WM in the contralateral hemisphere 
    - Brain stem
    - Cerebellum 
    - A coronal plane posterior to the temporal cortex
    '''

    dataLoc = '/Volumes/CCNC_3T_2/kcho/ccnc/GHR_project'
    subject_list = args.subjects

    # Make exclusion mask in order to exclude tracks
    # going towards posterior paths from the thalamus
    for subject in subject_list:
        roiLoc = os.path.join(dataLoc, subject, 'ROI')
        aseg_img = os.path.join(dataLoc, subject, 'FREESURFER/mri/aseg.mgz')

        brainStem = os.path.join(roiLoc, 'brain_stem.nii.gz')
        if not os.path.isfile(brainStem):
            brainStemExtract = fs.Binarize(out_type='nii.gz',
                    match = [16, 6, 7, 8, 45, 46, 47],
                    binary_file = brainStem,
                    aseg = aseg_img)
            brainStemExtract.run()

        for side in ['lh', 'rh']:
            thalamusROI = os.path.join(roiLoc, side+'_thalamus.nii.gz')
            LTC = os.path.join(roiLoc, side+'_LTC.nii.gz')
            MTC = os.path.join(roiLoc, side+'_MTC.nii.gz')
            TC_ROI = os.path.join(roiLoc, side+'_TC.nii.gz')

            OFC = os.path.join(roiLoc, side+'_OFC.nii.gz')
            LPFC = os.path.join(roiLoc, side+'_LPFC.nii.gz')
            MPFC = os.path.join(roiLoc, side+'_MPFC.nii.gz')
            FC_ROI = os.path.join(roiLoc, side+'_FC.nii.gz')

            post_TC_plane= os.path.join(roiLoc,side+'_post_thal_TC_excl_mask.nii.gz')
            post_thal_plane = os.path.join(roiLoc,side+'_post_thal_excl_mask.nii.gz')
            ant_thal_ex_mask =  os.path.join(roiLoc, side+'_ant_thal_excl_mask.nii.gz')
            wm_mask =  os.path.join(roiLoc, side+'_wm_mask.nii.gz')
            MNI_TC_mask_reg = os.path.join(roiLoc, side+'_MNI_TC_mask.nii.gz')

            if not os.path.isfile(MNI_TC_mask_reg):
                get_MNI_TC_mask_reg(dataLoc, subject, MNI_TC_mask_reg)

            if not os.path.isfile(TC_ROI):
                merge_LTC_MTC = fsl.MultiImageMaths(
                        in_file = LTC,
                        operand_files = [MTC],
                        op_string = '-add %s',
                        out_file = TC_ROI,
                        )
                merge_LTC_MTC.run()

            if not os.path.isfile(FC_ROI):
                merge_LPFC_MPFC_OFC = fsl.MultiImageMaths(
                        in_file = LPFC,
                        operand_files = [MPFC, OFC],
                        op_string = '-add %s -add %s',
                        out_file = FC_ROI,
                        )
                merge_LPFC_MPFC_OFC.run()


            if not os.path.isfile(post_thal_plane):
                thal_TC_posterior(thalamusROI, thalamusROI)

            if not os.path.isfile(post_TC_plane):
                thal_TC_posterior(thalamusROI, TC_ROI)

            if not os.path.isfile(ant_thal_ex_mask):
                if not os.path.isfile(wm_mask):
                    wmExtract = fs.Binarize(out_type='nii.gz',
                            in_file = aseg_img,
                            binary_file = wm_mask)
                    if args.side == 'lh':
                        # needs to put the contralateral hemisphere
                        wmExtract.inputs.match = [41]
                    else:
                        wmExtract.inputs.match = [2]
                    wmExtract.run()

                thal_anterior(thalamusROI, wm_mask, 
                        ant_thal_ex_mask, MNI_TC_mask_reg)


    # Dictionary for datasource
    info = dict(dwi=[['subject_id', 'data']],
                bvecs=[['subject_id', 'bvecs']],
                bvals=[['subject_id', 'bvals']],
                SEED=[['subject_id', 
                    ['THAL', 
                     'THAL',
                     'TC']]],
                STOP=[['subject_id', 
                    ['FC',
                     'TC',
                     'FC']]],
                avoid_mask = [['subject_id', 
                    ['post_thal_excl_mask',
                     'ant_thal_excl_mask',
                     'post_thal_TC_excl_mask' ]]],
                contra_wm = [['subject_id', 'wm_mask']],
                brainStem = [['subject_id', 'brain_stem']],
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
                                            SEED='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            STOP='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            avoid_mask='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            contra_wm='%s/ROI/{side}_%s.nii.gz'.format(side=get_opposite(airgs.side)),
                                            brainStem='%s/ROI/%s.nii.gz',
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



    
    # Merge exclusion ROIs
    # - brain stem
    # - contralateral hemisphere white matter 
    # - slice posterior to the thalamus 
    # - temporal lobe
    brainstem_plane = pe.MapNode(interface=fsl.ImageMaths(),
            name='brainstem_add_plane',
            iterfield = ['in_file2'])
    brainstem_plane.inputs.op_string = '-add %s'

    brainstem_plane_wm = pe.Node(interface=fsl.MultiImageMaths(), name='brainstem_add_plane_add_wm')
    brainstem_plane_wm.inputs.op_string = '-add %s'

    # Probabilistic tractography
    probtrackx = pe.MapNode(interface=fsl.ProbTrackX2(), 
            name='TC_FC_probtrackx_detailed',
            iterfield = ['seed', 'stop_mask', 'waypoints']
            )
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
    datasink.inputs.base_directory = os.path.abspath('/Volumes/CCNC_3T_2/kcho/ccnc/GHR_project/tractography')
    datasink.inputs.substitutions = [('_variable', 'variable'),
                                     ('_subject_id_', '')]

    # Workflow 
    dwiproc = pe.Workflow(name="Thal_TC_FC")
    dwiproc.base_dir = os.path.abspath('Processing')
    dwiproc.connect([
                        (infosource,datasource,[('subject_id', 'subject_id')]),
                        (datasource, brainstem_plane,[('brainStem', 'in_file')]),
                        (datasource, brainstem_plane,[('avoid_mask', 'in_file2')]),
                        (brainstem_plane, brainstem_plane_wm,[('out_file', 'in_file2')]),
                        (datasource, brainstem_plane_wm,[('wm_mask', 'in_file2')]),
                        (brainstem_plane_wm, datasink,[('out_file','exclusion_mask')]),
                        (brainstem_plane_wm, probtrackx,[('out_file', 'avoid_mp')]),
                        (datasource,probtrackx,[('SEED','seed'),
                                                   ('STOP','stop_mask'),
                                                   ('STOP','waypoints'),
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


def thal_TC_posterior(thalamusImg, TC_img=None):
    roiLoc = os.path.dirname(thalamusImg)
    side = os.path.basename(thalamusImg)[:2]

    # ROI load
    f_thal = nb.load(thalamusImg)
    data_thal = f_thal.get_data()

    f_TC = nb.load(TC_img)
    data_TC = f_TC.get_data()

    # find lowest z coordinate
    z_length = data_thal.shape[2]
    for sliceNumThal in range(z_length):
        if 1 in data_thal[:,:,sliceNumThal]:
            break

    z_length = data_thal.shape[2]
    for sliceNumTC in range(sliceNumThal):
        if 1 in data_TC[:,:,sliceNumTC]:
            break

    # lower the slice number the posterior the plane
    if sliceNumThal > sliceNumTC:
        sliceNum = sliceNumTC
    else:
        sliceNum = sliceNumThal

    newArray = np.zeros_like(data_thal)
    newArray[:,:,sliceNum] = 1

    post_thal_plane = os.path.join(roiLoc,side+'_post_thal_TC_excl_mask.nii.gz')
    nb.Nifti1Image(newArray, f_thal.affine).to_filename(post_thal_plane)


def get_MNI_TC_mask_reg(dataLoc, subject, MNI_TC_mask_reg):
    MNI_TC_mask = 'MNI_temporal_mask.nii.gz'
    roiLoc = os.path.join(dataLoc, subject, 'ROI')
    mni2subj = os.path.join(roiLoc, 'mni2subj.mat')
    subjBrain = os.path.join(dataLoc, subject, 'FREESURFER',
                        'mri','brain.nii.gz')
    atlasLoc = '/usr/local/fsl/data/atlases/MNI/MNI-maxprob-thr0-1mm.nii.gz'

    # Registration
    if not os.path.isfile(mni2subj):
        MNIreg = fsl.FLIRT(
                in_file = atlasLoc,
                reference = subjBrain,
                interp = 'nearestneighbour',
                out_matrix_file = mni2subj)
        MNIreg.run()

    # ROI extraction
    if not os.path.isfile(MNI_TC_mask):
        extractTC = fsl.ImageMaths(
                in_file = atlasLoc,
                op_string = '-thr 8 -uthr 8',
                out_file = MNI_TC_mask,
                )
        extractTC.run()

    # Apply registration
    TC_to_subj = fsl.ApplyXfm(
            in_file = MNI_TC_mask,
            reference = subjBrain,
            interp = 'nearestneighbour',
            in_matrix_file = mni2subj,
            out_file = MNI_TC_mask_reg)
    TC_to_subj.run()

def thal_anterior(thalamusROI, wm_mask, ant_thal_ex_mask, MNI_TC_mask_reg):
    roiLoc = os.path.dirname(thalamusROI)
    side = os.path.basename(thalamusROI)[:2]

    # ROI load
    f_thal = nb.load(thalamusROI)
    data_thal = f_thal.get_data()

    f_WM = nb.load(wm_mask)
    data_WM = f_WM.get_data()

    f_TC = nb.load(MNI_TC_mask_reg)
    data_TC = f_TC.get_data()

    # find highest z coordinate
    z_length = data_thal.shape[2]
    for sliceNumThal in range(z_length, 0, -1):
        if 1 in data_thal[:,:,sliceNumThal-1]:
            break

    # Zeroing the white matter mask posterior to the plane
    data_WM[:,:,:sliceNumThal] = 0
    data_WM_sub_TC = data_WM - data_TC
    data_WM_sub_TC[ data_WM_sub_TC < 0] = 0


    nb.Nifti1Image(data_WM_sub_TC, f_WM.affine).to_filename(ant_thal_ex_mask)


def get_opposite(side):
    if side=='lh':
        return 'rh'
    else:
        return 'lh'

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

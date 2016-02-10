#!/ccnc_bin/Canopy_env/User/bin/python
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nibabel as nb
import numpy as np
import argparse
import textwrap
import re
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

    dataLoc = '/home/kangik/bedpostCollect/kma'
    subject_list = args.subjects

    # Make exclusion mask in order to exclude tracks
    # going towards posterior paths from the thalamus

    # python way
    jobs = []

    for subject in subject_list:
        roiLoc = os.path.join(dataLoc, subject, 'ROI')
        bedpostLoc = os.path.join(dataLoc, subject, 'DTI.bedpostX')
        aseg_img = os.path.join(dataLoc, subject, 'FREESURFER/mri/aseg.mgz')
        aparc_img = os.path.join(dataLoc, subject, 'FREESURFER/mri/aparc+aseg.mgz')
        wmparc_img= os.path.join(dataLoc, subject, 'FREESURFER/mri/wmparc.mgz')
        reg_file = os.path.join(dataLoc, subject, 'registration/FREESURFERT1toNodif.mat')
        MNI_TC_mask_reg = os.path.join(roiLoc, 'MNI_TC_mask.nii.gz')
        MNI_OCC_mask_reg = os.path.join(roiLoc, 'MNI_OCC_mask.nii.gz')

        brainStem = os.path.join(roiLoc, 'brain_stem.nii.gz')
        if not os.path.isfile(brainStem):
            get_mask_from_fs(aseg_img, brainStem)

        for side in ['lh','rh']:
            wm_mask =  os.path.join(roiLoc, side+'_wm_mask.nii.gz')
            if not os.path.isfile(wm_mask):
                get_mask_from_fs(aseg_img, wm_mask)

        for side in ['lh', 'rh']:
            outDir = os.path.join(dataLoc, subject, 'tractography', side)
            try:
                os.mkdir(outDir)
            except:
                pass
            thalamusROI = os.path.join(roiLoc, side+'_thalamus.nii.gz')
            LTC = os.path.join(roiLoc, side+'_LTC.nii.gz')
            MTC = os.path.join(roiLoc, side+'_MTC.nii.gz')
            STG = os.path.join(roiLoc, side+'_STG.nii.gz')
            TC_ROI = os.path.join(roiLoc, side+'_TC.nii.gz')

            OFC = os.path.join(roiLoc, side+'_OFC.nii.gz')
            IFG = os.path.join(roiLoc, side+'_IFG.nii.gz')
            LPFC = os.path.join(roiLoc, side+'_LPFC.nii.gz')
            MPFC = os.path.join(roiLoc, side+'_MPFC.nii.gz')
            FC_ROI = os.path.join(roiLoc, side+'_FC.nii.gz')

            post_TC_plane= os.path.join(roiLoc,side+'_post_thal_TC_excl_mask.nii.gz')
            post_thal_plane = os.path.join(roiLoc,side+'_post_thal_excl_mask.nii.gz')
            ant_thal_ex_mask =  os.path.join(roiLoc, side+'_ant_thal_excl_mask.nii.gz')

            # Freesurfer exclusion mask
            wm_mask =  os.path.join(roiLoc, side+'_wm_mask.nii.gz')
            contra_wm = re.sub(side, get_opposite(side), wm_mask)
            fs_claustrum_wm = os.path.join(roiLoc, side+'_fs_claustrum_wm.nii.gz')
            fs_FC_wm = os.path.join(roiLoc, side+'_fs_FC_wm.nii.gz')
            fs_TC_wm = os.path.join(roiLoc, side+'_fs_TC_wm.nii.gz')
            fs_PC_wm = os.path.join(roiLoc, side+'_fs_PC_wm.nii.gz')
            fs_OCC_wm = os.path.join(roiLoc, side+'_fs_OCC_wm.nii.gz')


            for fsMask in [fs_FC_wm, fs_TC_wm, fs_PC_wm, fs_OCC_wm]:
                if not os.path.isfile(fsMask):
                    get_mask_from_fs(wmparc_img, fsMask)

            if not os.path.isfile(fs_claustrum_wm):
                get_mask_from_fs(aseg_img, fs_claustrum_wm)

            if not os.path.isfile(IFG):
                get_mask_from_fs(aparc_img, IFG)

            if not os.path.isfile(STG):
                get_mask_from_fs(aparc_img, STG)
            # Merged exclusion_masks
            TC_thal_ex_mask = os.path.join(roiLoc, side+'_TC_thal_ex_mask.nii.gz')
            FC_thal_ex_mask = os.path.join(roiLoc, side+'_FC_thal_ex_mask.nii.gz')
            FC_TC_ex_mask = os.path.join(roiLoc, side+'_FC_TC_ex_mask.nii.gz')
            IFG_STG_ex_mask = os.path.join(roiLoc, side+'_IFG_STG_ex_mask.nii.gz')


            if not os.path.isfile(MNI_TC_mask_reg):
                get_MNI_mask_reg(dataLoc, subject, MNI_TC_mask_reg)

            if not os.path.isfile(MNI_OCC_mask_reg):
                get_MNI_mask_reg(dataLoc, subject, MNI_OCC_mask_reg)

            if not os.path.isfile(TC_ROI):
                add_files(LTC, [MTC], TC_ROI)

            if not os.path.isfile(FC_ROI):
                add_files(LPFC, [MPFC, OFC], FC_ROI)

            if not os.path.isfile(post_thal_plane):
                thal_TC_posterior(thalamusROI, thalamusROI, post_thal_plane)

            if not os.path.isfile(post_TC_plane):
                thal_TC_posterior(thalamusROI, TC_ROI, post_TC_plane)

            if not os.path.isfile(ant_thal_ex_mask):
                thal_anterior(thalamusROI, wm_mask, 
                        ant_thal_ex_mask, MNI_TC_mask_reg)


            # Exclusion masks
            if not os.path.isfile(TC_thal_ex_mask):
                add_files(brainStem,
                        [contra_wm, ant_thal_ex_mask, post_TC_plane, fs_OCC_wm, fs_PC_wm],
                        TC_thal_ex_mask)

            if not os.path.isfile(FC_thal_ex_mask):
                add_files(brainStem,
                        [contra_wm, post_thal_plane, fs_TC_wm, fs_OCC_wm, fs_PC_wm],
                        FC_thal_ex_mask)
    
            if not os.path.isfile(FC_TC_ex_mask):
                add_files(brainStem,
                        [contra_wm, post_TC_plane, MNI_OCC_mask_reg, fs_OCC_wm, fs_PC_wm, thalamusROI],
                        FC_TC_ex_mask)

            if not os.path.isfile(IFG_STG_ex_mask):
                add_files(brainStem,
                        [contra_wm, post_TC_plane, MNI_OCC_mask_reg, fs_claustrum_wm, fs_OCC_wm, fs_PC_wm, thalamusROI],
                        FC_TC_ex_mask)



            FC_TC_prob_command = '/usr/local/fsl/bin/probtrackx2 \
                    -x {thalamusSeed} \
                    -l \
                    --onewaycondition \
                    -c 0.2 \
                    -S 2000 \
                    --steplength=0.5 \
                    -P 5000 \
                    --fibthresh=0.01 \
                    --distthresh=0.0 \
                    --sampvox=0.0 \
                    --xfm={freesurferT1toNodif} \
                    --avoid={exclusionMask} \
                    --stop={targetMask} \
                    --forcedir \
                    --opd \
                    -s {bedpostLoc}/merged \
                    -m {bedpostLoc}/nodif_brain_mask \
                    --dir={outDir} \
                    --waypoints={targetMask} \
                    --waycond=AND'.format(thalamusSeed = STG,
                            freesurferT1toNodif = reg_file,
                            exclusionMask = FC_TC_ex_mask,
                            bedpostLoc = bedpostLoc,
                            outDir = outDir,
                            targetMask = IFG)

            jobs.append(re.sub('\s+',' ',FC_TC_prob_command))

                             

    for job in jobs:
        print job
        print


    # Dictionary for datasource
    #info = dict(dwi=[['subject_id', 'data']],
                #bvecs=[['subject_id', 'bvecs']],
                #bvals=[['subject_id', 'bvals']],
                #SEED=[['subject_id', 
                    #['thalamus', 
                     #'thalamus',
                     #'TC']]],
                #STOP=[['subject_id', 
                    #['FC',
                     #'TC',
                     #'FC']]],
                #avoid_mask = [['subject_id', 
                    #['FC_thal_ex_mask',
                     #'TC_thal_ex_mask',
                     #'FC_TC_ex_mask']]],
                #thsample = [['subject_id',
                    #['merged_th1samples','merged_th2samples']]],
                #phsample = [['subject_id',
                    #['merged_ph1samples','merged_ph2samples']]],
                #fsample = [['subject_id',
                    #['merged_f1samples','merged_f2samples']]],
                #matrix = [['subject_id','FREESURFERT1toNodif.mat']],
                #bet_mask = [['subject_id','nodif_brain_mask']],
                #fsLoc = [['subject_id','freesurfer']],
                #aseg = [['subject_id','aseg']],
                #aparc_aseg= [['subject_id','aparc+aseg']],
                #)

    ## Subject name list setting into identity interface
    #infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                         #name="infosource")
    #infosource.iterables = ('subject_id', subject_list)


    ## Data source setting
    #datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                                   #outfields=info.keys()),
                         #name = 'datasource')
    #datasource.inputs.template = "%s/%s"
    #datasource.inputs.base_directory = os.path.abspath(dataLoc)
    #datasource.inputs.field_template = dict(dwi='%s/dti/%s.nii.gz',
                                            #bvecs='%s/dti/%s',
                                            #bvals='%s/dti/%s',
                                            #SEED='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            #STOP='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            #avoid_mask='%s/ROI/{side}_%s.nii.gz'.format(side=args.side),
                                            #matrix='%s/registration/%s',
                                            #bet_mask='%s/dti/%s.nii.gz',
                                            #thsample='%s/DTI.bedpostX/%s.nii.gz',
                                            #fsample='%s/DTI.bedpostX/%s.nii.gz',
                                            #phsample='%s/DTI.bedpostX/%s.nii.gz',
                                            #fsLoc='%s/%s',
                                            #aseg='%s/FREESURFER/mri/%s.mgz',
                                            #aparc_aseg='%s/FREESURFER/mri/%s.mgz',
                                            #)
    #datasource.inputs.template_args = info
    #datasource.inputs.sort_filelist = True



    ## Probabilistic tractography
    #probtrackx = pe.MapNode(interface=fsl.ProbTrackX2(), 
            #name='TC_FC_probtrackx_detailed',
            #iterfield = ['seed', 'stop_mask', 'waypoints', 'avoid_mp']
            #)
    #probtrackx.inputs.onewaycondition= True

    ##Below are the default options
    ##probtrackx.inputs.waycond = 'AND'
    ##probtrackx.inputs.c_thresh = 0.2
    ##probtrackx.inputs.n_steps = 2000
    ##probtrackx.inputs.step_length = 0.5
    ##probtrackx.inputs.n_samples = 5000
    ##probtrackx.inputs.opd = True
    ##probtrackx.inputs.os2t = True
    ##probtrackx.inputs.loop_check = True


    ## Data sink
    #datasink = pe.Node(interface=nio.DataSink(),name='datasink')
    #datasink.inputs.base_directory = os.path.abspath('/Volumes/CCNC_3T_2/kcho/ccnc/GHR_project/tractography')
    #datasink.inputs.substitutions = [('_variable', 'variable'),
                                     #('_subject_id_', '')]

    ## Workflow 
    #dwiproc = pe.Workflow(name="Thal_TC_FC")
    #dwiproc.base_dir = os.path.abspath('Processing')
    #dwiproc.connect([
                        #(infosource,datasource,[('subject_id', 'subject_id')]),
                        #(datasource,probtrackx,[('SEED','seed'),
                                                   #('STOP','stop_mask'),
                                                   #('STOP','waypoints'),
                                                   #('avoid_mask','avoid_mp'),
                                                   #('bet_mask','mask'),
                                                   #('phsample','phsamples'),
                                                   #('fsample','fsamples'),
                                                   #('thsample','thsamples'),
                                                   #('matrix','xfm'),
                                                   #]),
                        #(probtrackx,datasink,[('fdt_paths','probtrackx.@fdt_paths'),
                            #('log', 'probtrackx.@log'),
                            #('particle_files', 'probtrackx.@particle_files'),
                            #('targets', 'probtrackx.@targets'),
                            #('way_total', 'probtrackx.@way_total'),
                            #])
                    #])

    ## Parallel processing
    #dwiproc.run(plugin='MultiProc', plugin_args={'n_procs' : 8})


def thal_TC_posterior(thalamusImg, TC_img, outFile):
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

    nb.Nifti1Image(newArray, f_thal.affine).to_filename(outFile)


def get_MNI_mask_reg(dataLoc, subject, outROI):
    MNI_mask = os.path.basename(outROI)[3:]
    roiLoc = os.path.join(dataLoc, subject, 'ROI')
    mni2subj = os.path.join(roiLoc, 'mni2subj.mat')
    subjBrain = os.path.join(dataLoc, subject, 'FREESURFER', 
            'mri','brain.nii.gz')
    atlasLoc = '/usr/local/fsl/data/atlases/MNI/MNI-maxprob-thr0-1mm.nii.gz'

    if outROI.endswith('TC_mask.nii.gz'):
        op_string = '-thr 8 -uthr 8'
    elif outROI.endswith('OCC_mask.nii.gz'):
        op_string = '-thr 5 -uthr 5'



    # Registration
    if not os.path.isfile(mni2subj):
        MNIreg = fsl.FLIRT(
                in_file = atlasLoc,
                reference = subjBrain,
                interp = 'nearestneighbour',
                out_matrix_file = mni2subj)
        print 'MNI <--> subject flirt is running'
        MNIreg.run()

    # ROI extraction
    if not os.path.isfile(MNI_mask):
        extractTC = fsl.ImageMaths(
                in_file = atlasLoc,
                op_string = op_string,
                out_file = MNI_mask,
                )
        extractTC.run()

    # Apply registration
    TC_to_subj = fsl.ApplyXfm(
            in_file = MNI_mask,
            reference = subjBrain,
            interp = 'nearestneighbour',
            in_matrix_file = mni2subj,
            out_file = outROI)
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

def get_mask_from_fs(aseg_img, binaryROI):
    if binaryROI.endswith('brain_stem.nii.gz'):
        match = [16, 6, 7, 8, 45, 46, 47]
    elif binaryROI.endswith('lh_IFG.nii.gz'):
        match = [1018, 1019, 1020]
    elif binaryROI.endswith('rh_IFG.nii.gz'):
        match = [2018, 2019, 2020]

    elif binaryROI.endswith('lh_STG.nii.gz'):
        match = [1030]
    elif binaryROI.endswith('rh_STG.nii.gz'):
        match = [2030]

    elif binaryROI.endswith('lh_wm_mask.nii.gz'):
        match = [2]
    elif binaryROI.endswith('rh_wm_mask.nii.gz'):
        match = [41]

    elif binaryROI.endswith('lh_fs_claustrum_wm.nii.gz'):
        match = [138]
    elif binaryROI.endswith('rh_fs_claustrum_wm.nii.gz'):
        match = [139]

    elif binaryROI.endswith('lh_fs_FC_wm.nii.gz'):
        match = [3002, 3026, 3028, 3020, 3027, 3032, 3018, 3019, 3014, 3012]
    elif binaryROI.endswith('rh_fs_FC_wm.nii.gz'):
        match = [4002, 4026, 4028, 4020, 4027, 4032, 4018, 4019, 4014, 4012]

    elif binaryROI.endswith('lh_fs_FC_wm.nii.gz'):
        match = [3002, 3026, 3028, 3020, 3027, 3032, 3018, 3019, 3014, 3012]
    elif binaryROI.endswith('rh_fs_FC_wm.nii.gz'):
        match = [4002, 4026, 4028, 4020, 4027, 4032, 4018, 4019, 4014, 4012]

    elif binaryROI.endswith('lh_fs_TC_wm.nii.gz'):
        match = [3034, 3030, 3001, 3009, 3015, 3033, 3006, 3016, 3007]
    elif binaryROI.endswith('rh_fs_TC_wm.nii.gz'):
        match = [4034, 4030, 4001, 4009, 4015, 4033, 4006, 4016, 4007]

    elif binaryROI.endswith('lh_fs_PC_wm.nii.gz'):
        match = [3008, 3031, 3025, 3023, 3010, 3029]
    elif binaryROI.endswith('rh_fs_PC_wm.nii.gz'):
        match = [4008, 4031, 4025, 4023, 4010, 4029]

    elif binaryROI.endswith('lh_fs_OCC_wm.nii.gz'):
        match = [3021, 3013, 3011, 3005]
    elif binaryROI.endswith('rh_fs_OCC_wm.nii.gz'):
        match = [4021, 4013, 4011, 4005]


    binarize = fs.Binarize(out_type='nii.gz',
            match = match,
            binary_file = binaryROI,
            in_file = aseg_img)
    binarize.run()

def add_files(in_file, in_file2, out_file):
    op_string = ' '.join(['-add %s'] * len(in_file2))
    merge = fsl.MultiImageMaths(
            in_file = in_file,
            operand_files = in_file2,
            op_string = op_string,
            out_file = out_file,
            )
    merge.run()

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

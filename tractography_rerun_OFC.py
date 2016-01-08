#!/usr/bin/env python
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import argparse
import textwrap
import os                                    # system functions

def main(args):
    print args.subjects
    subject_list = args.subjects

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
    datasource.inputs.base_directory = os.path.abspath('/Volumes/CCNC_3T_2/kcho/ccnc/GHR_project')
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
    datasink.inputs.base_directory = os.path.abspath('/Volumes/CCNC_3T_2/kcho/ccnc/GHR_project/OFC_tractography')
    datasink.inputs.substitutions = [('_variable', 'variable'),
                                     ('_subject_id_', 'subject_id')]

    ''' Workflow '''
    dwiproc = pe.Workflow(name="dwiproc_OFC")
    dwiproc.base_dir = os.path.abspath('tractography_OFC')
    dwiproc.connect([
                        (infosource,datasource,[('subject_id', 'subject_id')]),
                        (datasource, insulaExtract, [('aparc_aseg', 'in_file')]),
                        (datasource, brainStemExtract, [('aseg', 'in_file')]),
                        (insulaExtract,cortexStopRoiMaker,[('binary_file', 'in_file')]),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            {codeName} : Search files with user defined extensions 
            ========================================
            eg) {codeName} -e 'dcm|ima' -i /Users/kevin/NOR04_CKI
                Search dicom files in /Users/kevin/NOR04_CKI
            eg) {codeName} -c -e 'dcm|ima' -i /Users/kevin/NOR04_CKI
                Count dicom files in each directory under input 
            '''.format(codeName=os.path.basename(__file__))))
    parser.add_argument(
        '-s', '--subjects',
        help='subject list',
        nargs='+')
    parser.add_argument(
        '-side', '--side',
        help='side')
    args = parser.parse_args()

    main(args)
    #print sys.argv

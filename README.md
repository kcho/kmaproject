# Probabilistic tractography project

It runs three probabilistic tractography runs between
- temporal cortex
- frontal cortex
- thalamus

It requires FSL, Freesurfer and nipype.

Each subject directory under the dataLoc must be 
structured as the format below

```
dataLoc
├── subjectDir1
│   ├── FREESURFER
│   ├── DTI.bedpostX
│   ├── registration
│   │    └──FREESURFERT1toNodif.mat
│   │       (from flirt between Freesurfer Brain.mgz
│   │       and DTI nodif)
│   └── ROI
│
├── subjectDir2
```

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

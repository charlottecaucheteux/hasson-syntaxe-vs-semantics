import json
import os
from glob import glob

import nibabel as nib
import numpy as np
from narratives.exclude_scans import exclude_scan
from natsort import natsorted

from . import paths


def read_gifti(gifti_fn):
    gii = nib.load(gifti_fn)
    data = np.vstack([da.data[np.newaxis, :] for da in gii.darrays])
    return data


def get_bold(gii_fname, subject, exclude=True, afni_dir=paths.afni_dir):

    # Load scans to exclude
    scan_exclude = json.load(open(paths.scan_exclude_path))

    # Load bold responses
    subj_data = None
    bold_fns = natsorted(glob(str(afni_dir / subject / "func" / gii_fname)))
    assert len(bold_fns), str(afni_dir / subject / "func" / gii_fname)
    for bold_fn in bold_fns:

        if exclude and exclude_scan(bold_fn, scan_exclude):
            print(f"Excluding {os.path.basename(bold_fn)}!")
            continue
        subj_data = read_gifti(bold_fn)

    return subj_data

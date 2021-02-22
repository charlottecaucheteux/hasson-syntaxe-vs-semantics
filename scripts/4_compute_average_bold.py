import numpy as np
from tqdm.notebook import tqdm

from src import paths
from src.get_bold import get_bold
from src.task_dataset import get_task_df
from pathlib import Path

def compute_mean_bold(hemi, df_task, save_file, space="fsaverage6"):
    bold = {}
    counts = {}
    for i, row in tqdm(df_task.iterrows()):
        if i % 100 == 0:
            print("i", i)
        # Load bold responses
        gii_fname = f"{row.subject}_task-{row.bold_task}_*space-{space}_hemi-{hemi}_desc-clean.func.gii"
        try:
            subj_data = get_bold(
                gii_fname, row.subject, exclude=True, afni_dir=paths.afni_dir_nosmooth
            )
        except AssertionError as e:
            print("Assertion error")
            continue
        if subj_data is None:
            continue
        subj_data = subj_data[row.onset :, :]
        if row.audio_task not in bold:
            bold[row.audio_task] = subj_data
            counts[row.audio_task] = 0
        else:
            nr, nc = np.stack([subj_data.shape, bold[row.audio_task].shape]).min(0)
            bold[row.audio_task] = bold[row.audio_task][:nr, :nc]
            bold[row.audio_task] += subj_data[:nr, :nc]
            counts[row.audio_task] += 1
    mean_bolds = {k: bold[k] / counts[k] for k in bold.keys()}
    np.save(save_file, mean_bolds)

            
def compute_median_bold_one_task(hemi, df_task, task, save_path, space = "fsaverage6"):
    print(task, hemi)
    bold = []
    errors = []
    df_task = df_task.query("audio_task==@task")
    print(len(df_task))
    for i, row in tqdm(df_task.iterrows()):
        if i%100==0:
            print("i", i)
        # Load bold responses
        gii_fname = f"{row.subject}_task-{row.bold_task}_*space-{space}_hemi-{hemi}_desc-clean.func.gii"
        try:
            subj_data = get_bold(
            gii_fname, row.subject, exclude=True, afni_dir=paths.afni_dir_nosmooth
        )
        except AssertionError as e:
            print("Assertion error")
            errors.append(gii_fname)
            continue
        if subj_data is None:
            errors.append(gii_fname)
            continue        
        bold.append(subj_data[row.onset:])
    if len(bold)==0:
        med_bolds = {"bold": None, "count":0, "errors":errors}
    else:
        nr, nc = np.stack([b.shape for b in bold]).min(0)
        print(f"Cutting to {nr}, {nc}")
        count = len(bold)
        bold = [b[:nr, :nc] for b in bold]
        bold = np.median(np.stack(bold), axis=0)
        med_bolds = {"bold": bold , "count":count, "task":task, "errors":errors} 
    
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    np.save(save_path, med_bolds)
    
    
if __name__=="__main__":
    df_task = get_task_df()
    tasks = [p.parent.name for p in list(paths.gentle_path.glob("*/align.csv"))]
    for task in tasks:
        for hemi in ["L", "R"]:
            save_path = Path(f"data/bold/median_bold/{hemi}") / f"{task}.npy"
            if not save_path.is_file():
                compute_median_bold_one_task(hemi, df_task, task, save_path)
            
    # Combine
    print("Combining bolds")
    for hemi in ["L", "R"]:
        bold = {"bold" : {}, "count": {}, "task": {}, "errors": {}}
        for task in tasks:
            save_path = Path(f"data/bold/median_bold/{hemi}") / f"{task}.npy"
            bold_task = np.load(save_path, allow_pickle=True).item()
            for k, v in bold_task.items():
                bold[k][task] = v
        np.save(f"data/bold/median_bold_concat_tasks_{hemi}.npy", bold)
        

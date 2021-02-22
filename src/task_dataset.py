import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths


class Dataset:
    """Task/subtask/group/subject iterator for simplicity"""

    def __init__(self, base_dir):

        # Get metadata for all subjects for a given task

        base_dir = Path(base_dir)
        with open(base_dir / "code" / "task_meta.json") as f:
            self.task_meta = json.load(f)

        # Skip 'schema' task for simplicity
        skip_tasks = ["notthefalllongscram", "notthefallshortscram", "schema"]

        for skip in skip_tasks:
            self.task_meta.pop(skip)

        # Skip 'notthefall' scramble and 'schema' tasks for simplicity
        self.task_id = 0
        self.subtask_id = 0
        self.group_id = 0
        self.subject_id = 0

    def __repr__(self):
        print("task_id %i " % self.task_id)
        print("subtask_id %i" % self.subtask_id)
        print("group_id %i" % self.group_id)

    def __iter__(self):
        return self

    def __next__(
        self,
    ):

        # task
        task_keys = list(self.task_meta.keys())
        task = task_keys[self.task_id]

        # subtask
        subtasks = ["slumlord", "reach"] if task == "slumlordreach" else [task]
        subtask = subtasks[self.subtask_id]

        # groups
        if task == "milkyway":
            groups = ["original", "vodka", "synonyms"]
        # elif task == 'prettymouth':
        #    groups = ['affair', 'paranoia']
        else:
            groups = [None]
        group = groups[self.group_id]

        stim_label = task + group if group else task

        # subjects
        subjects = sorted(self.task_meta[task].keys())
        subject = subjects[self.subject_id]

        # next iter

        def next():
            # update iterator
            self.subject_id += 1
            if self.subject_id == len(subjects):
                self.subject_id = 0
                self.group_id += 1
            if self.group_id == len(groups):
                self.group_id = 0
                self.subtask_id += 1
            if self.subtask_id == len(subtasks):
                self.subtask_id = 0
                self.task_id += 1
            if self.task_id == len(task_keys):
                raise StopIteration

        if group and group != self.task_meta[subtask][subject]["condition"]:
            next()
            return self.__next__()

        next()

        return task, subtask, group, subject, stim_label


def get_task_df(task_exclude=["notthefalllongscram", "notthefallshortscram", "schema"]):
    # Matching between task and subjects
    df = pd.read_csv(paths.base_dir / "participants.tsv", sep="\t")
    df = df.astype("str")
    dataset = []
    for i, row in df.iterrows():
        for task, condition, comprehension in zip(
            row.task.split(","), row.condition.split(","), row.comprehension.split(",")
        ):
            if task == "milkyway":
                task == task + condition
            if task in task_exclude:
                continue
            if comprehension != "n/a":
                comprehension = float(comprehension)
                if "shapes" in task:
                    comprehension /= 10
            else:
                comprehension = np.nan
            dataset.append(
                {
                    "audio_task": task,
                    "subject": row.participant_id,
                    "comprehension": comprehension,
                }
            )
    dataset = pd.DataFrame(dataset)

    # Task info
    checked_tasks = get_checked_tasks()

    # Merge
    dataset = pd.merge(dataset, checked_tasks, on="audio_task", how="inner")

    return dataset


def get_checked_tasks():
    """
    Tasks checked by hand - report onset of the paper.
    """
    tasks = [
        p.parent.name for p in list(Path(paths.checked_gentle_path).glob("*/align.csv"))
    ]
    tasks = {k: {} for k in tasks}

    for key in tasks.keys():
        if key in [
            "milkywayoriginal",
            "milkywaysynonyms",
            "milkywayvodka",
        ]:
            tasks[key]["bold_task"] = "milkyway"
        else:
            tasks[key]["bold_task"] = key

    # Set onsets for some tasks
    for key in [
        "21styear",
        "milkywayoriginal",
        "milkywaysynonyms",
        "milkywayvodka",
        "prettymouth",
        "pieman",
    ]:
        tasks[key]["onset"] = 0
    for key in ["piemanpni", "bronx", "black", "forgot"]:
        tasks[key]["onset"] = 8
    for key in [
        "slumlordreach",
        "shapessocial",
        "shapesphysical",
        "sherlock",
        "merlin",
        "notthefallintact",
    ]:
        tasks[key]["onset"] = 3
    for key in ["lucy"]:
        tasks[key]["onset"] = 1  # not aligned with text
    for key in ["tunnel"]:
        tasks[key]["onset"] = 2
    checked_tasks = {k: v for k, v in tasks.items() if "onset" in v}
    checked_tasks = pd.DataFrame(checked_tasks).T.reset_index()
    checked_tasks = checked_tasks.rename(columns={"index": "audio_task"})
    return checked_tasks


def create_checked_stimuli():
    """Save to a new directory checked stimuli"""
    tasks_with_issues = ["notthefallintact", "prettymouth", "merlin"]
    new_starts = [[25.8], [21], [29, 29.15]]
    # new_gentle = Path(str(paths.init_gentle_path) + "_checked")
    new_gentle = paths.checked_gentle_path
    tasks = [p.parent.name for p in list(Path(paths.gentle_path).glob("*/align.csv"))]
    for task in tasks:
        print(task)
        df = pd.read_csv(paths.gentle_path / task / "align.csv", header=None)
        Path(new_gentle / task).mkdir(exist_ok=True)
        df.to_csv(new_gentle / task / "align.csv", header=None, index=False)
    for task, new_vals in zip(tasks_with_issues, new_starts):
        df = pd.read_csv(paths.gentle_path / task / "align.csv", header=None)
        for i, val in enumerate(new_vals):
            df.iloc[i, 2] = val
            df.iloc[i, 3] = val + 0.05
        df.to_csv(new_gentle / task / "align.csv", header=None, index=False)

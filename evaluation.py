from datetime import time
import os 
import pandas as pd
import numpy as np
from haven import haven_utils as hu
from src import models
import time as t
def count_params():
    base_path = os.path.join(os.getcwd(), 'CovidSeg', 'save')
    experiment_folders = os.listdir(base_path)
    for exp in experiment_folders:
        if len(os.listdir(os.path.join(base_path, exp))) == 8: # Already evaluate.
            continue
        os.system('python CovidSeg/count_params.py --exp_id {}'.format(exp))

def evaluate():
    base_path = os.path.join(os.getcwd(), 'CovidSeg', 'save')
    experiment_folders = os.listdir(base_path)
    for exp in experiment_folders:
        if len(os.listdir(os.path.join(base_path, exp))) == 8: # Already evaluate.
            continue
        os.system('python CovidSeg/test.py --exp_id {}'.format(exp))
    

def summary():
    base_path = os.path.join(os.getcwd(), 'CovidSeg', 'save')
    experiment_folders = os.listdir(base_path)
    valid = 'valid'
    summaries = []
    for exp in experiment_folders:
        score_path = os.path.join(base_path, exp, valid, 'score.csv')
        df = pd.read_csv(score_path)
        iou_0 = df["iou_group0"].values.tolist()[0]
        iou_1 = df["iou_group1"].values.tolist()[0]
        iou_2 = df["iou_group2"].values.tolist()[0]
        summaries.append([exp, iou_0, iou_1, iou_2, sum([iou_0, iou_1, iou_2]) / 3])

    pd.DataFrame(summaries).to_csv(os.path.join(base_path, 'result.csv'))

if __name__ == '__main__':
    evaluate()
    t.sleep(1)
    summary()
    # count_params()
    pass
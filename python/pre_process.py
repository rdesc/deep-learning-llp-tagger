import numpy as np
import seaborn as sns

import pandas as pd

def pre_process(data):
    filter_clus_pt = [col for col in data if col.startswith("clus_pt")]
    print(data[filter_clus_pt])
    data[filter_clus_pt] = data[filter_clus_pt].sub(data["clus_pt_0"], axis='index')

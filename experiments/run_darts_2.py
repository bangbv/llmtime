import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from data.small_context import get_datasets_2
from data.serialize import SerializerSettings
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from models.utils import grid_iter
from models.gaussian_process import get_gp_predictions_data
from models.darts import get_TCN_predictions_data, get_NHITS_predictions_data, get_NBEATS_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.darts import get_arima_predictions_data
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
# openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

# Specify the hyperparameter grid for each model
gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

model_hypers = {
    'gpt-4': {'model': 'gpt-4', **gpt4_hypers},
}

# Specify the function to get predictions for each model
model_predict_fns = {
    'gpt-4': get_llmtime_predictions_data,
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'gpt-4']])

# Specify the output directory for saving results
output_dir = 'outputs/darts'
os.makedirs(output_dir, exist_ok=True)

datasets = get_datasets_2()
for dsname,data in datasets.items():
    train, test = data
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    
    # N-HiTS, TCN and N-BEATS require training and can be slow. Skip them if you want quick results.
    
    for model in ['text-davinci-003', 'gpt-4', 'gp', 'arima', 'N-HiTS', 'TCN', 'N-BEATS']:
        if model in out_dict:
            print(f"Skipping {dsname} {model}")
            continue
        else:
            print(f"Starting {dsname} {model}")
            hypers = list(grid_iter(model_hypers[model]))
            print(f"hypers: {hypers}")
        parallel = True if is_gpt(model) else False
        print(f"parallel: {parallel}")
        num_samples = 20 if is_gpt(model) else 100
        print(f"num_samples: {num_samples}")
        try:
            preds = get_autotuned_predictions_data(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel)
            out_dict[model] = preds
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(out_dict,f)

    print(f"Finished {dsname}")
    
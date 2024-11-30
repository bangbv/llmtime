import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from data.small_context import get_datasets_2
from data.serialize import SerializerSettings
from models.validation_likelihood_tuning import get_autotuned_predictions_data_2
from models.utils import grid_iter
from models.utils import print_debug, my_print
from models.gaussian_process import get_gp_predictions_data
from models.darts import get_TCN_predictions_data, get_NHITS_predictions_data, get_NBEATS_predictions_data
from models.llmtime import get_llmtime_predictions_data, get_llmtime_predictions_data_2
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

llama_hypers = dict(
    temp=1.0,
    alpha=0.99,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, time_sep=',', bit_sep='', plus_sign='', minus_sign='-', signed=True),
)

model_hypers = {
    'llama-7b': {'model': 'llama-7b', **llama_hypers},
    'gpt-4': {'model': 'gpt-4', **gpt4_hypers},
}

# Specify the function to get predictions for each model
model_predict_fns = {
    'llama-7b': get_llmtime_predictions_data,
    'gpt-4': get_llmtime_predictions_data_2,
}

def is_gpt(model):
    return any([x in model for x in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'gpt-4']])

# Specify the output directory for saving results
output_dir = 'outputs/darts'
os.makedirs(output_dir, exist_ok=True)

datasets = get_datasets_2(3, 0.1)
for dsname,data in datasets.items():
    log_debug = False
    train, test = data
    print_debug(my_print, "train size", len(train), log_debug)
    print_debug(my_print, "train data", train, log_debug)
    print_debug(my_print, "test size", len(test), log_debug)
    print_debug(my_print, "test data", test, log_debug)
    if os.path.exists(f'{output_dir}/{dsname}.pkl'):
        with open(f'{output_dir}/{dsname}.pkl','rb') as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}
    
    # N-HiTS, TCN and N-BEATS require training and can be slow. Skip them if you want quick results.
    
    for model in ['gpt-4']:
        if model in out_dict:
            print(f"Skipping {dsname} {model}")
            continue
        else:
            print(f"Starting {dsname} {model}")
            hypers = list(grid_iter(model_hypers[model]))
            print(f"hypers: {hypers}")
        parallel = True if is_gpt(model) else False
        if log_debug : print(f"parallel: {parallel}")
        print_debug(my_print, "parallel", parallel, log_debug)
        num_samples = 20 if is_gpt(model) else 100
        print_debug(my_print, "num_samples", num_samples, log_debug)
        try:
            preds = get_autotuned_predictions_data_2(train, test, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=parallel, log_debug=log_debug)
            print_debug(my_print, "preds", preds, log_debug)
            out_dict[model] = preds
        except Exception as e:
            print(f"Failed {dsname} {model}")
            print(e)
            continue
        with open(f'{output_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(out_dict,f)

    print(f"Finished {dsname}")
    
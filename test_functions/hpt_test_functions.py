# Code modified from https://github.com/tennisonliu/LLAMBO/tree/master

import os
import json
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

from functools import partial

import torch

TASK_MAP = {
    'breast': ['classification', 'accuracy'],
    'digits': ['classification', 'accuracy'],
    'wine': ['classification', 'accuracy'],
    'diabetes': ['regression', 'neg_mean_squared_error'],
}

BEST_PERFORMANCE = {
    "wine": {
        "global_max": 1.0,
        "global_min": 0.4444444444444444
    },
    "breast": {
        "global_max": 0.9824561403508771,
        "global_min": 0.5877192982456141
    },
    "digits": {
        "global_max": 0.8888888888888888,
        "global_min": 0.08055555555555556
    },
    "diabetes": {
        "global_max": 0.8389739213304408,
        "global_min": 0.5817586934044268
    }
}

DATASETS = ['breast', 'digits', 'wine', 'diabetes']

MODELS = ['RandomForest', 'DecisionTree', 'SVM', 'AdaBoost', 'MLPSGD']

def get_bayesmark_func(model_name, task_type):
    # https://github.com/uber/bayesmark/blob/master/bayesmark/sklearn_funcs.py
    assert model_name in MODELS, f'Unknown model name: {model_name}'
    assert task_type in ['classification', 'regression']
    if model_name == 'RandomForest':
        if task_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            return partial(RandomForestClassifier, n_estimators=10, max_leaf_nodes=None, random_state=0)    # following Bayesmark implementation
        elif task_type == 'regression':
            from sklearn.ensemble import RandomForestRegressor
            return partial(RandomForestRegressor, n_estimators=10, max_leaf_nodes=None, random_state=0)
        
    if model_name == 'DecisionTree':
        if task_type == 'classification':
            from sklearn.tree import DecisionTreeClassifier
            return partial(DecisionTreeClassifier, max_leaf_nodes=None, random_state=0)
        elif task_type == 'regression':
            from sklearn.tree import DecisionTreeRegressor
            return partial(DecisionTreeRegressor, max_leaf_nodes=None, random_state=0)

    if model_name == 'SVM':
        if task_type == 'classification':
            from sklearn.svm import SVC
            return partial(SVC, kernel='rbf', probability=True, random_state=0)
        elif task_type == 'regression':
            from sklearn.svm import SVR
            return partial(SVR, kernel='rbf')
        
    if model_name == 'AdaBoost':
        if task_type == 'classification':
            from sklearn.ensemble import AdaBoostClassifier
            return partial(AdaBoostClassifier, random_state=0)
        elif task_type == 'regression':
            from sklearn.ensemble import AdaBoostRegressor
            return partial(AdaBoostRegressor, random_state=0)

    if model_name == 'MLPSGD':
        if task_type == 'classification':
            from sklearn.neural_network import MLPClassifier
            return partial(MLPClassifier, solver='sgd', early_stopping=True, max_iter=40,
                           learning_rate='invscaling', nesterovs_momentum=True, random_state=0)
        elif task_type == 'regression':
            from sklearn.neural_network import MLPRegressor
            return partial(MLPRegressor, solver='sgd', activation='tanh', early_stopping=True, max_iter=40,
                           learning_rate='invscaling', nesterovs_momentum=True, random_state=0)

class BayesmarkExpRunner:
    def __init__(self, task_context, dataset):
        self.model = task_context['model']
        self.task = task_context['task']
        self.metric = task_context['metric']
        self.dataset = dataset
        self.hyperparameter_constraints = task_context['hyperparameter_constraints']
        self.bbox_func = get_bayesmark_func(self.model, self.task)
    
    def generate_initialization(self, n_samples, seed):
        '''
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: list of dictionaries, each dictionary is a point to be evaluated
        '''

        # Read from fixed initialization points (all baselines see same init points)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_fpath = os.path.join(current_dir, 'bayesmark_configs', f'{self.model}/{seed}.json')
        init_configs = pd.read_json(json_fpath).head(n_samples)
        init_configs = init_configs.to_dict(orient='records')

        assert len(init_configs) == n_samples

        return init_configs
        
    def evaluate_point(self, candidate_config):
        '''
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        fvals can contain an arbitrary number of items, but also must contain 'score' (which is what LLAMBO optimizer tries to optimize)
        fvals = {
            'score': float,                     -> 'score' is what the LLAMBO optimizer tries to optimize
            'generalization_score': float
        }
        '''
        X_train, X_test, y_train, y_test = self.dataset['train_x'], self.dataset['test_x'], self.dataset['train_y'], self.dataset['test_y']

        for hyperparam, value in candidate_config.items():
            if self.hyperparameter_constraints[hyperparam][0] == 'int':
                candidate_config[hyperparam] = int(value)

        if self.task == 'regression':
            mean_ = np.mean(y_train)
            std_ = np.std(y_train)
            y_train = (y_train - mean_) / std_
            y_test = (y_test - mean_) / std_

        model = self.bbox_func(**candidate_config)
        scorer = get_scorer(self.metric)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            S = cross_val_score(model, X_train, y_train, scoring=scorer, cv=5)
        cv_score = np.mean(S)
        
        model = self.bbox_func(**candidate_config)  
        model.fit(X_train, y_train)
        generalization_score = scorer(model, X_test, y_test)

        return candidate_config, {'score': cv_score, 'generalization_score': generalization_score}

def load_task_context(model, dataset, X_train, y_train):
    # Describe task context
    task_context = {}
    task_context['model'] = model
    task_context['task'] = TASK_MAP[dataset][0]
    task_context['tot_feats'] = X_train.shape[1]
    task_context['cat_feats'] = 0       # bayesmark datasets only have numerical features
    task_context['num_feats'] = X_train.shape[1]
    task_context['n_classes'] = len(np.unique(y_train))
    task_context['metric'] = TASK_MAP[dataset][1]
    task_context['lower_is_better'] = True if 'neg' in task_context['metric'] else False
    task_context['num_samples'] = X_train.shape[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_fpath = os.path.join(current_dir, 'bayesmark.json')
    with open(json_fpath, 'r') as f:
        task_context['hyperparameter_constraints'] = json.load(f)[model]
    return task_context

def load_dataset(dataset_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_fpath = os.path.join(current_dir, 'bayesmark_data', f'{dataset_name}.pickle')
    with open(pickle_fpath, 'rb') as f:
        data = pickle.load(f)
    return data

def create_bayesmark_class(model, dataset):
    data = load_dataset(dataset)
    task_context = load_task_context(model, dataset, data['train_x'], data['train_y'])
    class BayesmarkProblem(BayesmarkExpRunner):
        dim = len(task_context['hyperparameter_constraints'].keys())
        name = f"hpt_{dataset}_{model}"
        
        def __init__(self):
            self.input_order = {}
            self.lower_bounds, self.upper_bounds = [], []
            for i, hyperparam in enumerate(task_context['hyperparameter_constraints'].keys()):
                self.input_order[i] = hyperparam
                constraints = task_context['hyperparameter_constraints'][hyperparam][-1]
                self.lower_bounds.append(constraints[0])
                self.upper_bounds.append(constraints[1])
            self.bounds = torch.tensor([self.lower_bounds, self.upper_bounds])
            if task_context["lower_is_better"]:
                self._optimal_value = BEST_PERFORMANCE[dataset]["global_min"]
            else:
                self._optimal_value = - BEST_PERFORMANCE[dataset]["global_max"]
            super().__init__(task_context, data)

        def map_X(self, X):
            # Map vector to configs
            return {hyperparam: X[i] for i, hyperparam in self.input_order.items()}
        
        def map_configs(self, configs):
            # Map configs to vector
            return np.array([configs[hyperparam] for hyperparam in self.input_order.values()])

        def _evaluate_true(self, X):
            if X.ndim == 1:
                X = X.unsqueeze(0)
            # map candidate configuration to hyperparameters
            result = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
            for i in range(X.shape[0]):
                input_vector = X[i].detach().cpu().numpy()
                config = self.map_X(input_vector)
                _, output = self.evaluate_point(config)
                result[i] = - output['score'] # negative because we assume minimization
            return result

        def __call__(self, X):
            return self._evaluate_true(X)

    return BayesmarkProblem
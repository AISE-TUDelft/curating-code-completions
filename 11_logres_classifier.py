import os, math, pickle, wandb, pandas as pd, numpy as np
while not os.getcwd().endswith('-analysis'): os.chdir('..')
from pprint import pprint
from dataclasses import dataclass
from typing import Any, Callable

import sys 
TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'

from sklearn.linear_model import LogisticRegression
from notebooks.paper.util import (
    GridSearch,
    search,
    common_evaluation,
    get_contextual_query_data,
    get_nontextual_features,
    fold_iterator,
)

# NOTE: Feature retrieval functions given a query 
def _shared_features(query):
    ''' Maintaining this so it's clearer which function adds what '''
    return [
        math.log(1 + query.time_since_last_completion),
        math.log(1 + query.get_document_length()),
        math.log(1 + query.get_offset()),
        query.get_offset_as_percentage(),
        *query.get_document_language_vector(),          # 5-24
    ]

def copilot(query):
    ''' Features used in reverse-engineering Copilot, 
        except those that depend on a pre-existing filter implementation '''

    return [
        *_shared_features(query),
        # we don't have a previous filter label
        int(query.get_whitespace_after_cursor()),
        # time since last label should be very close to time_since_last_completion
        math.log(1 + query.get_prefix_last_line_length()),
        math.log(1 + query.get_prefix_trimmed_last_line_length()),
        *query.get_prefix_last_character_vector(),
        *query.get_trimmed_prefix_last_character_vector()
    ]

def tr_copilot(query): 
    ''' Same as above, without the last character vector '''
    return [
        *_shared_features(query),
        int(query.get_whitespace_after_cursor()),
        math.log(1 + query.get_prefix_last_line_length()),
        math.log(1 + query.get_prefix_trimmed_last_line_length()),
        *query.get_prefix_last_character_vector(),
        # *query.get_trimmed_prefix_last_character_vector()
    ]

def ide_and_copilot(query):
    ''' Same as copilot_features, with below IDE features '''
    return [
        1 if query.ide == 'jetbrains' else 0, 
        1 if query.ide == 'vsc' else 0,
        *copilot(query),
    ]

def nontextual(query) -> list:
    ''' Get the features that could otherwise not be extracted from the context alone, 
        This is identical to `get_nontextual_features` from util.py '''
    return [
        *_shared_features(query),                       # 2-24
        1 if query.ide == 'jetbrains' else 0,           # 0
        1 if query.ide == 'vsc' else 0,                 # 1
    ]


@dataclass 
class Hyperparameters(GridSearch):

    query_data : Any 
    project    : str        = 'code-completion-analysis'
    subdir     : str        = '11_logres'

    n_folds    : int        = 5 if not TEST else 1 
    seed       : int        = 42

    partition  : str        = search(['biased', 'subclasses', 'classes', 'unbalanced'])
    model      : Any        = lambda: LogisticRegression(max_iter=1000, random_state=42)

    # Different feature combinations
    feature_fn : Callable   = search([copilot, tr_copilot, ide_and_copilot, nontextual])

    @property 
    def out_dir(self):
        return os.path.join('notebooks/paper/results', self.subdir, self.partition)
    
    @property
    def name(self):
        return '-'.join([
            'test' if TEST else '',
            self.subdir,
            self.partition,
            self.feature_fn.__name__,
        ])

def train(h: Hyperparameters, fold, name):

    X_train, y_train = np.array([np.array(h.feature_fn(q)) for q in fold['X_train']]), fold['y_train']

    model : LogisticRegression = h.model() 
    model.fit(X_train, y_train)

    X_eval, y_eval = fold['X_eval'], fold['y_eval']
    X_test, y_test = h.query_data['unbalanced']['X_test'], h.query_data['unbalanced']['y_test']

    eval_metrics = common_evaluation(
        lambda X_queries: np.array([np.array(h.feature_fn(q)) for q in X_queries]),
        lambda X: model.predict(X),
        X_eval,
        y_eval,
    )

    test_metrics = common_evaluation(
        lambda X_queries: np.array([np.array(h.feature_fn(q)) for q in X_queries]),
        lambda X: model.predict(X),
        X_test,
        y_test,
    )

    score = {'n features': X_train.shape[1]}
    score.update({f'eval {k}': v for k, v in eval_metrics.items()})
    score.update({f'test {k}': v for k, v in test_metrics.items()})

    print(f'{name}: {X_train.shape[0]} training samples with {X_train.shape[1]} features. {eval_metrics["accuracy"]:.1f}% eval acc. {test_metrics["accuracy"]:.1f}% test acc.')

    save_dir = os.path.join(h.out_dir, 'model')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    with open(os.path.join(save_dir, f'{name}.pkl'), 'wb') as f:
        pickle.dump(model, f)

    return score 


if __name__ == '__main__':

    results = {} 
    hyperparameters = Hyperparameters( query_data = get_contextual_query_data(debug=False) ) # debug=False as LogRes takes like 10s to train

    for h in hyperparameters:
        pprint(h)

        folds = fold_iterator(h.query_data[h.partition], seed=h.seed)
        for i, fold in enumerate(folds):
            if i >= h.n_folds: break 

            fold_name = f'{h.name}-{i}'
            print(f'\t{h.name} \n\t\033[1m{fold_name}\033[0m')

            # using wandb for tracking all my runs, seems a bit overkill for logres but whatever
            wandb.init(project=h.project, group=h.name, name=fold_name)

            score = train(h, fold, fold_name)

            wandb.log(score)
            wandb.finish()
            pprint(score)

            results[fold_name] = score

    pprint(results)
    pd.DataFrame(results).T.to_csv(f'{h.out_dir}/results.csv')

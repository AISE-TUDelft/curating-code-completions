import os, shutil, tqdm, torch, wandb, pandas as pd
while not os.getcwd().endswith('-analysis'): os.chdir('..')
from pprint import pprint
from dataclasses import dataclass
from datasets import DatasetDict
from typing import Any 

import sys 
TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'

from notebooks.paper.util import (
    common_evaluation, 
    get_compute_metrics,
    get_trainer,
    get_contextual_query_data, 
    get_dataset, 
    encode_data,
    set_all_seeds, 
    fold_iterator,
    GridSearch,
    search,
    MyPipeline,
)

@dataclass 
class Hyperparameters(GridSearch):

    query_data      : Any 

    project         : str = 'code-completion-analysis'
    subdir          : str = '12_codeberta'

    model_dir       : str = 'huggingface/CodeBERTa-small-v1'
    partition       : str = 'biased'
    out_dir         : str = os.path.join('notebooks/paper/results', subdir, model_dir)

    n_folds         : int = 5
    seed            : int = 42
    load_best_model_at_end : bool = True

    num_train_epochs : int = 3 
    learning_rate    : float = search([2e-5, 1e-5, 5e-5])
    batch_size       : int = 16

    @property
    def dataset(self): 
        return encode_data(self.query_data[self.partition])
   
    @property
    def name(self):
        return '-'.join([
            self.subdir,
            self.partition,
            f'{self.learning_rate}lr',
            'test' if TEST else ''
        ])


from transformers import RobertaForSequenceClassification
def train(h: Hyperparameters, fold, name):

    set_all_seeds(h.seed)

    model = RobertaForSequenceClassification.from_pretrained(h.model_dir)

    # we need to specify compute_metrics here as the trainer's evaluation relies on the original queries
    train_ds, eval_ds, queries = fold
    dataset = {'train': train_ds, 'eval': eval_ds}

    compute_metrics = get_compute_metrics(queries)

    trainer = get_trainer(model, dataset, h, compute_metrics)
    trainer.train()
    trainer.save_model(os.path.join(h.out_dir, 'model', name))

    shutil.rmtree(os.path.join(h.out_dir, 'train'))

    return evaluate(trainer, h)


def evaluate(trainer, h: Hyperparameters): 
    ''' Evaluate the best model'''

    eval_metrics = trainer.evaluate()

    pipe = MyPipeline(
        device=0, task='text-classification',
        model=trainer.model, incl_features=False # for telemetry 
    )

    X_test, y_test = h.query_data['unbalanced']['X_test'], h.query_data['unbalanced']['y_test']
    test_metrics = common_evaluation(
        lambda X_queries: get_dataset(X_queries).to_iterable_dataset(),
        lambda X: torch.cat([x for x in tqdm.tqdm(pipe(X), total=len(X_test), desc='sequential test')]),
        X_test,
        y_test,
    )

    score = {'n features': 512}
    score.update({f'eval {k}': v for k, v in eval_metrics.items()})
    score.update({f'test {k}': v for k, v in test_metrics.items()})

    return score


if __name__ == '__main__': 

    results = {} 
    hyperparameters = Hyperparameters(
        query_data=get_contextual_query_data(debug=TEST)
    )

    for h in hyperparameters:
        pprint(h)

        folds = fold_iterator(h.dataset, h.query_data[h.partition], seed=h.seed)
        for i, fold in enumerate(folds):
            if i >= h.n_folds: break 

            name = f'{h.name}-{i}'
            group = (h.name + 'test') if TEST else h.name
            wandb.init(project=h.project, group=h.name, name=name)

            score = train(h, fold, name)

            wandb.log(score)
            wandb.finish()
            results[name] = score
    
    pprint(results)
    pd.DataFrame(results).T.to_csv(f'{h.out_dir}/results.csv')

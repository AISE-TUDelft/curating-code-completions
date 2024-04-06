import os, shutil, tqdm, torch, wandb, pandas as pd
while not os.getcwd().endswith('-analysis'): os.chdir('..')
from pprint import pprint
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
    set_all_seeds, 
    fold_iterator,
    GridSearch,
    search,
    MyPipeline,
    tokenize_joint_sample as util_tokenize_joint_sample,
)

from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')

def tokenize_prefix(samples, max_length = tokenizer.model_max_length): 
    ''' Tokenize samples by tokenizing and left-truncating prefix, and padding to max length '''

    tokenizer.truncation_side = 'left'
    return tokenizer(samples['prefix'], padding='max_length', truncation=True, return_tensors='pt')

def tokenize_suffix(samples, max_length = tokenizer.model_max_length):
    ''' Tokenize samples by tokenizing and right-truncating suffix, and padding to max length '''

    tokenizer.truncation_side = 'right'
    return tokenizer(samples['suffix'], max_length = max_length, padding='max_length', truncation=True, return_tensors='pt')

def tokenize_joint_sample(*args, **kwargs):
    ''' We maintain this in util to not have copies of the same function,
        and to avoid circular dependencies '''
    return util_tokenize_joint_sample(*args, **kwargs)

from dataclasses import dataclass


@dataclass 
class Hyperparameters(GridSearch):

    query_data      : Any 

    project         : str = 'code-completion-analysis'
    subdir          : str = 'tokenisation'

    model_dir       : str = 'huggingface/CodeBERTa-small-v1'
    partition       : str = 'biased'
    out_dir         : str = os.path.join('notebooks/paper/results', subdir, model_dir)

    n_folds         : int = 5
    seed            : int = 42
    load_best_model_at_end : bool = True

    num_train_epochs : int = 3 
    learning_rate    : float = 2e-5
    batch_size       : int = 16

    tok_fn           : Any = search([tokenize_prefix, tokenize_suffix, tokenize_joint_sample])

    @property
    def dataset(self): 
        data = self.query_data[self.partition]
        data = DatasetDict({
            'train': get_dataset(data['X_train'], data['y_train']), 
            'eval': get_dataset(data['X_eval'], data['y_eval'])
        })
        return data.map(self.tok_fn, batched=True, batch_size=1, num_proc=32)

    @property
    def name(self):
        return '-'.join([
            self.subdir,
            self.tok_fn.__name__.split('_')[1],
            'test' if TEST else ''
        ])


from transformers import RobertaForSequenceClassification
def train(h: Hyperparameters, fold):

    set_all_seeds(h.seed)

    model = RobertaForSequenceClassification.from_pretrained(h.model_dir)

    # we need to specify compute_metrics here as the trainer's evaluation relies on the original queries
    train_ds, eval_ds, queries = fold
    dataset = {'train': train_ds, 'eval': eval_ds}

    compute_metrics = get_compute_metrics(queries)

    trainer = get_trainer(model, dataset, h, compute_metrics)
    trainer.train()
    trainer.save_model(os.path.join(h.out_dir, 'model'))

    shutil.rmtree(os.path.join(h.out_dir, 'train'))

    return evaluate(trainer, h)


def evaluate(trainer, h: Hyperparameters): 
    ''' Evaluate the best model'''

    eval_metrics = trainer.evaluate()

    pipe = MyPipeline(
        device=0, task='text-classification',
        model=trainer.model, preprocess_fn=h.tok_fn,
        incl_features=False # for telemetry 
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

            score = train(h, fold)

            wandb.log(score)
            wandb.finish()
            results[name] = score
    
    pprint(results)
    pd.DataFrame(results).T.to_csv(f'{h.out_dir}/results.csv')

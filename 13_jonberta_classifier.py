import os, shutil, tqdm, torch, wandb, pandas as pd
while not os.getcwd().endswith('-analysis'): os.chdir('..')
from pprint import pprint
from dataclasses import dataclass
from datasets import DatasetDict
from typing import Any 

import sys 
TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'

from safetensors import safe_open
from transformers import RobertaForSequenceClassification, AutoConfig
from notebooks.paper.modeling_jonberta import JonbertaForSequenceClassification, add_features_to_model
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
    print_new_params,
)

@dataclass 
class HeadConfig(GridSearch):

    num_telemetry_features :int = 26
    add_head              :bool = True
    add_dense             :bool = search([True, False])
    add_proj              :bool = search([True, False])
    reinit_head           :bool = search([True, False])

    @property
    def name(self):
        return '-'.join([
            'HEAD',
            'dense' if self.add_dense else '',
            'proj' if self.add_proj else '',
            'reinit' if self.reinit_head else '',
        ])

@dataclass 
class AttnConfig(GridSearch):

    num_telemetry_features  :int = 26

    add_feature_embeddings :bool = True 
    feature_hidden_size     :int = num_telemetry_features * 8 
    feature_dropout_prob  :float = 0.1
    add_feature_bias       :bool = True

    add_self_attn          :bool = True
    self_attn_layers  :list[int] = search([[i,j,k] for i in range(6) for j in range(6) for k in range(6) if i < j < k])
    # self_attn_layers  :list[int] = search([[i, j] for j in range(6) for i in range(6) if i < j])
    # self_attn_layers  :list[int] = search([[i] for i in range(6)])

    @property
    def name(self):
        return '-'.join([
            'ATTN',
            f'{self.feature_hidden_size}C_f',
            f'{self.self_attn_layers}L',
        ])


@dataclass 
class Hyperparameters(GridSearch):

    query_data      : Any 

    project         : str = 'code-completion-analysis-layers'
    subdir          : str = '13_jonberta'

    model_dir       : str = 'huggingface/CodeBERTa-small-v1'
    # model_dir       : str = 'notebooks/paper/results/12_codeberta/huggingface/CodeBERTa-small-v1/model/12_codeberta-biased-2e-05lr--0'
    partition       : str = 'biased'
    out_dir         : str = os.path.join('notebooks/paper/results', subdir, model_dir.split('/')[-1])

    n_folds         : int = 5 
    seed            : int = 42
    load_best_model_at_end : bool = True

    num_train_epochs : int      = 3 
    learning_rate    : float    = 2e-5
    batch_size       : int      = 16

    # NOTE: Temporary configs that didn't run because someone killed my job on ronaldo
    # config         : HeadConfig = search([
    #     HeadConfig(add_dense=False, add_proj=False, reinit_head=True),
    #     HeadConfig(add_dense=False, add_proj=False, reinit_head=False)
    # ])
    # config         : HeadConfig = search([config for config in HeadConfig()])
    config         : AttnConfig = search([config for config in AttnConfig()])

    @property
    def dataset(self): 
        return encode_data(self.query_data[self.partition])
   
    @property
    def name(self):
        return '-'.join([
            'TEST' if TEST else '',
            self.subdir,
            self.partition,
            self.model_dir.split('/')[-1],
            f'({self.config.name})',
            f'{self.learning_rate}lr',
        ])

def get_model(h: Hyperparameters):

    config = AutoConfig.from_pretrained(h.model_dir)
    config.update(h.config.__dict__)

    if isinstance(h.config, HeadConfig):
        model = JonbertaForSequenceClassification.from_pretrained(h.model_dir, config=config)
        add_features_to_model(model, config)
        return model 
    
    elif isinstance(h.config, AttnConfig):
        model = JonbertaForSequenceClassification(config)
        if 'huggingface' in h.model_dir:
            hf_model = RobertaForSequenceClassification.from_pretrained(h.model_dir)
            model.load_state_dict(hf_model.state_dict(), strict=False)
            del hf_model
        else:
            state_dict = {} 
            with safe_open(os.path.join(h.model_dir, 'model.safetensors'), framework='pt') as f: 
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            new_layers = model.load_state_dict(state_dict, strict=False)
            print(new_layers)

        print_new_params(model)
        return model 

    else: 
        raise NotImplementedError(f'Unknown config type: {type(h.config)}')

def train(h: Hyperparameters, fold, name):

    train_dir = os.path.join(h.out_dir, 'train', name)
    save_dir = os.path.join(h.out_dir, 'model', name)
    set_all_seeds(h.seed)

    model = get_model(h)

    train_ds, eval_ds, queries = fold
    compute_metrics = get_compute_metrics(queries)
    dataset = {'train': train_ds, 'eval': eval_ds}

    trainer = get_trainer(model, dataset, h, compute_metrics, train_dir)
    trainer.train()
    trainer.save_model(save_dir)

    shutil.rmtree(os.path.join(train_dir))

    return evaluate(trainer, h)


def evaluate(trainer, h: Hyperparameters): 
    ''' Evaluate the best model'''

    eval_metrics = trainer.evaluate()

    pipe = MyPipeline(
        device=0, task='text-classification',
        model=trainer.model, incl_features=True # for telemetry 
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
        query_data  = get_contextual_query_data(debug=TEST),
        n_folds     = 1 if TEST else 5,
    )

    for h in hyperparameters:
        pprint(h)

        folds = fold_iterator(h.dataset, h.query_data[h.partition], seed=h.seed)
        for i, fold in enumerate(folds):
            if i >= h.n_folds: break 

            fold_name = f'{h.name}-{i}'
            print(f'\t{h.name} \n\t\033[1m{fold_name}\033[0m')

            wandb.init(project=h.project, group=h.name, name=fold_name)

            score = train(h, fold, fold_name)

            wandb.log(score)
            wandb.finish()
            results[fold_name] = score
    
    pprint(results)
    pd.DataFrame(results).T.to_csv(f'{h.out_dir}/results.csv')

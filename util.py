from __future__ import annotations
from itertools import product
from dataclasses import dataclass
from typing import List, Any, Generator, Tuple

import os, random, torch, time, math, tqdm, numpy as np
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import set_seed, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
from REDACTED import Query, Queries, CopilotQuery


@dataclass
class Search:
    ''' TODO: prohibit importing of this '''
    params : List[Any]

    def __iter__(self):
        ''' iterates over this hyperparam's values '''
        return iter(self.params)
    
    def __len__(self):
        return len(self.params)
    
    def __str__(self): 
        return str(self.params)

def search(params: List[Any]) -> Any:
    return Search(params=params)

@dataclass 
class GridSearch:
    ''' extend this class with your hyperparameter dataclass,
        to allow for iterating over its parameters as a grid search'''
    # Example given below.

    def __post_init__(self):
        if len(self.__dimensions) == 0: return 

        to_print = '\n\t'.join(f'{k}: \t{v}' for k, v in self.__dimensions.items())
        print(f'''
        Search {len(self.__dimensions)} dimensions, total {len(self.__combinations)} combinations
        \033[90m{to_print}\033[0m
        ''')   

    def __iter__(self) -> Generator[GridSearch, None, None]:
        for combination in self.__combinations: 
            yield self.__class__(**{**self.__dict__, **combination})

    @property 
    def __dimensions(self) -> dict[str, Search]:
        ''' returns all search params, and their search space '''
        return {k: v for k, v in self.__dict__.items() \
                if isinstance(v, Search)}

    @property 
    def __combinations(self) -> List[dict[str, Any]]:
        ''' returns all combinations of search params'''
        search_params = self.__dimensions
        if len(search_params) == 0: return [self.__dict__]

        combinations = list(product(*search_params.values()))
        return [{k: v[i] for i, k in enumerate(search_params.keys())}\
                 for v in combinations]

@dataclass
class Hyperparams(GridSearch):

    model_dir  : str = 'hugggingface/CodeBERTa-small-v1'
    config_dir : str = 'hugggingface/CodeBERTa-small-v1'

    num_telemetry_features : int    = 26
    feature_layers         : int    = search([1, 2, 3])
    add_cross_attn         : bool   = search([True, False])

    batch_size             : int    = 16
    num_train_epochs       : int    = 3
    learning_rate          : float  = 2e-5

    ## TODO: test
    # - no params
    # - single param
    # - multiple params
    # - search 1 param
    # - search no params 
    # - search multiple params
    # - recursive structures!! 


def set_all_seeds(seed=42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.distributed.barrier()

# NOTE: PREPROCESSING
def get_nontextual_features(query: CopilotQuery) -> list:
    ''' Get the features that could otherwise not be extracted from the context alone '''
    return [
        1 if query.ide == 'jetbrains' else 0,           # 0
        1 if query.ide == 'vsc' else 0,                 # 1
        math.log(1 + query.time_since_last_completion), # 2
        math.log(1 + query.get_document_length()),      # 3
        math.log(1 + query.get_offset()),               # 4
        query.get_offset_as_percentage(),               # 5
        *query.get_document_language_vector(),          # 6-25
    ]

def get_dataset(queries: list[CopilotQuery], labels: list[int] = None) -> Dataset:
    ''' Create a HF Dataset from a list of queries and optional labels '''

    if labels is not None: assert len(labels) == len(queries), 'Labels must be same length as queries'

    def my_gen():
        for i, query in enumerate(queries):
            dictionary = {
                'prefix': query.prefix,
                'suffix': query.suffix,
                'encoder_hidden_states': get_nontextual_features(query)
            }
            if labels is not None: dictionary['label'] = int(labels[i])
            yield dictionary
    
    return Dataset.from_generator(my_gen)

tokenizer = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')
def tokenize_joint_sample(sample, max_suffix_tokens=128):
    ''' For a single sample, tokenize prefix and suffix, separating by </s> sep token. 
        Set max_suffix_tokens to maximal amount of suffix to include, when it exists. '''

    max_length = tokenizer.model_max_length # 512 

    # figure out how many suffix tokens we have (128 max)
    tokenizer.truncation_side = 'right'
    suffix = tokenizer(sample['suffix'], padding='do_not_pad', truncation=True, return_tensors='pt',
                          max_length = max_suffix_tokens + 1) # to accomodate removal of <s>

    n_suffix_tokens = len(suffix['input_ids'][0]) - 1

    tokenizer.truncation_side = 'left'
    prefix = tokenizer(sample['prefix'], padding='do_not_pad', truncation=True, return_tensors='pt',
                       max_length = max_length - n_suffix_tokens)

    n_prefix_tokens = len(prefix['input_ids'][0])
    tokenizer.truncation_side = 'right'
    suffix = tokenizer(sample['suffix'], padding='max_length', truncation=True, return_tensors='pt',
                       max_length = max_length - n_prefix_tokens + 1) # to accomodate removal of <s>
    
    suffix['input_ids'] = suffix['input_ids'][:, 1:]
    suffix['attention_mask'] = suffix['attention_mask'][:, 1:]

    sample.update({k: torch.cat((prefix[k], suffix[k]), dim=1) for k in prefix})
    return sample

def encode_data(partition, s=slice(None,None,None)): 
    ''' takes a partition query_dict and returns a HF Dataset.
        Optionally takes a slice for using a subset in testing '''
    dataset = DatasetDict({
        'train': get_dataset(partition['X_train'][s], partition['y_train'][s]), 
        'eval': get_dataset(partition['X_eval'][s], partition['y_eval'][s])
    })
    return dataset.map(tokenize_joint_sample, batched=True, batch_size=1, 
                       num_proc=32 if torch.cuda.is_available() else 4, \
                       desc='tokenizing',
                       remove_columns=['prefix', 'suffix'])

# NOTE: EVALUATION
# import evaluate as hf_evaluate
# hf_accuracy = hf_evaluate.load('accuracy')
# hf_precision = hf_evaluate.load('precision')
# hf_recall = hf_evaluate.load('recall')
# hf_f1 = hf_evaluate.load('f1')
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def timeit(func, n_samples) -> Tuple[Any, float]: 
    ''' Compute time (seconds) per sample, given a function and n_samples '''
    t0 = time.time()
    res = func()
    return res, (time.time() - t0) / n_samples

def common_evaluation(preprocess_fn, predict_fn, X_queries: list[Query], y, return_preds=False):
    ''' Common evaluation computing accuracy, false negatives, false positives, and time per sample 
        Params:
            preprocess_fn: function that takes in a list of queries and returns a list of preprocessed queries
            predict_fn: function that takes in a list of preprocessed queries and returns a list of predictions
            X_queries: list of queries
            y: list of labels
        Returns:
            dict of 'accuracy', 'fn', 'fp', 'time', 
            as well as accuracy for each subclass: 'man acc', 'man rej', 'auto acc', 'auto rej'
    '''

    X, prep_time = timeit(lambda: preprocess_fn(X_queries), len(y))
    y_pred, pred_time = timeit(lambda: predict_fn(X), len(y))

    # Necessary to avoid flaky np fn/fp computation
    y, y_pred = np.array(y), np.array(y_pred)

    accuracy = 100 * accuracy_score(y, y_pred)
    fn = 100* np.sum(np.logical_and(y_pred == 0, y == 1)) / np.sum(y == 1)
    fp = 100* np.sum(np.logical_and(y_pred == 1, y == 0)) / np.sum(y == 0)

    # Now, need to figure these out per subclass
    # so, split based on accepted/rejected manual/automatic 
    # and compute accuracy for each of these subsets 
    man_acc, man_rej, auto_acc, auto_rej = [], [], [], []
    for query, pred, label in zip(X_queries, y_pred, y):
        if query.is_manually_invoked():
            if query.is_accepted():
                man_acc.append(pred == label)
            else: 
                man_rej.append(pred == label)
        else:
            if query.is_accepted():
                auto_acc.append(pred == label)
            else: 
                auto_rej.append(pred == label)

    man_acc = 100 * np.sum(man_acc) / len(man_acc)
    man_rej = 100 * np.sum(man_rej) / len(man_rej)
    auto_acc = 100 * np.sum(auto_acc) / len(auto_acc)
    auto_rej = 100 * np.sum(auto_rej) / len(auto_rej)

    # print(f'man acc: {len(man_acc)} samples \tman rej: {len(man_rej)} \tauto acc: {len(auto_acc)} \tauto rej: {len(auto_rej)}')
    score = {
        # **hf_precision.compute(predictions=y_pred, references=y),
        # **hf_recall.compute(predictions=y_pred, references=y),
        # **hf_f1.compute(predictions=y_pred, references=y),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'accuracy': accuracy, 
        'fn': fn, 
        'fp': fp, 
        'time': prep_time + pred_time, 
        'man acc': man_acc,
        'man rej': man_rej,
        'auto acc': auto_acc,
        'auto rej': auto_rej,
        'macro avg': sum([man_acc, man_rej, auto_acc, auto_rej]) / 4,
        'prep time': prep_time, 
        'pred time': pred_time,
    }

    if return_preds: 
        return score, y_pred
    return score

from transformers.pipelines import TextClassificationPipeline
class MyPipeline(TextClassificationPipeline):
    ''' oh yeah custom pipeline because of the custom tokenisation!
        how convenient huggingface ill hug your face extra hard next time i see you '''
    
    def __init__(self, *args, incl_features=True, preprocess_fn=tokenize_joint_sample, **kwargs):
        super().__init__(*args, **kwargs)
        self.incl_features = incl_features
        self.preprocess_fn = preprocess_fn

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {} 
        if 'preprocess_fn' in kwargs: 
            preprocess_kwargs['preprocess_fn'] = kwargs.pop('preprocess_fn')
        return preprocess_kwargs, {}, {} 
    
    def preprocess(self, inputs, preprocess_fn=None):
        inputs = preprocess_fn(inputs) if self.preprocess_fn is None else self.preprocess_fn(inputs)
        if 'prefix' in inputs: del inputs['prefix']
        if 'suffix' in inputs: del inputs['suffix']
        # given that pipeline is used in sequential eval, we neeed to add a batch dimension for the model to not throw a tantrum
        if self.incl_features:
            inputs['encoder_hidden_states'] = torch.tensor(inputs['encoder_hidden_states'], dtype=torch.float32).unsqueeze(0)
        elif 'encoder_hidden_states' in inputs: 
            del inputs['encoder_hidden_states']
        return inputs
    
    def _forward(self, model_inputs):
        return self.model(**model_inputs)
    
    def postprocess(self, model_outputs):
        return model_outputs.logits.argmax(-1)

def evaluate(model, query_data, hyperparams, s=slice(None,None,None), incl_features=True): 
    ''' Evaluates a model on given query_data, and hyperparameters containing: 
        - partition
        - num_telemetry_features 
    '''
    partition = hyperparams.partition

    model.eval()
    pipe = MyPipeline(
        device=0,
        task='text-classification', 
        model=model, preprocess_fn=tokenize_joint_sample,
        incl_features=incl_features, # False for original CodeBERTa-small-v1
    )

    X_eval, y_eval = query_data[partition]['X_eval'][s], query_data[partition]['y_eval'][s]
    eval_metrics = common_evaluation(
        lambda X_queries: get_dataset(X_queries).to_iterable_dataset(),
        lambda X: torch.cat([x for x in tqdm.tqdm(pipe(X), total=len(X_eval), desc='sequential eval')]),
        X_eval,
        y_eval,
    )

    X_test, y_test = query_data['unbalanced']['X_test'][s], query_data['unbalanced']['y_test'][s]
    test_metrics = common_evaluation(
        lambda X_queries: get_dataset(X_queries).to_iterable_dataset(),
        lambda X: torch.cat([x for x in tqdm.tqdm(pipe(X), total=len(X_test), desc='sequential test')]),
        X_test,
        y_test,
    )

    results = {'n features': 512 + hyperparams.num_telemetry_features}
    results.update({f'eval {k}': v for k, v in eval_metrics.items()})
    results.update({f'test {k}': v for k, v in test_metrics.items()})

    return results

# NOTE: Training
def get_compute_metrics(query_data):
    ''' Returns function to be used in HF Trainer for evaluation, 
        Depends on the subclasses in the dataset, hence you need to pass query_data '''

    def compute_metrics(eval_pred): 
        if len([e for e in eval_pred]) == 3:
            logits, labels, inputs = eval_pred
        else: 
            logits, labels = eval_pred 
            inputs = None

        predictions = np.argmax(logits, axis=1)

        metric_dict = hf_accuracy.compute(predictions=predictions, references=labels)
        metric_dict.update(hf_precision.compute(predictions=predictions, references=labels))
        metric_dict.update(hf_recall.compute(predictions=predictions, references=labels))
        metric_dict.update(hf_f1.compute(predictions=predictions, references=labels))

        # TODO: I'm really not proud of the below code, but time is of the essence
        if query_data is not None: 
            queries = query_data['X_eval']

            assert len(predictions) == len(queries), f'{len(predictions)} != {len(queries)}'

            n_per_subclass = {'man acc': (0, 0), 'man rej': (0, 0), 'auto acc': (0, 0), 'auto rej': (0, 0)}
            for query, prediction, label in zip(queries, predictions, labels): 
                scl = 'man ' if query.is_manually_invoked() else 'auto '
                scl += 'acc' if query.is_accepted() else 'rej'
                n_per_subclass[scl] = (n_per_subclass[scl][0] + int(prediction == label), n_per_subclass[scl][1] + 1)
                
                # sanity chekc
                if label == 1: assert 'man' in scl or scl == 'auto acc'
                else: assert scl == 'auto rej'

            acc_per_subclass = {k: 100 * v[0] / v[1] if v[1] != 0 else 0 \
                    for k, v in n_per_subclass.items()}
            metric_dict.update(acc_per_subclass)
            metric_dict.update({'macro avg': np.mean(list(acc_per_subclass.values()))})
            metric_dict.update({f'n {k}': v[1] for k, v in n_per_subclass.items()})
            metric_dict.update({f'c {k}': v[0] for k, v in n_per_subclass.items()})

        fn = 100* np.sum((predictions == 0) & (labels == 1)) / np.sum(labels == 1)
        fp = 100* np.sum((predictions == 1) & (labels == 0)) / np.sum(labels == 0)

        metric_dict.update({'fp': fp, 'fn': fn})
        return metric_dict
    return compute_metrics

def get_trainer(model, encoded_dataset, h, compute_metrics, train_dir=None):
    ''' Constructs a Trainer from a model, dataset, and hyperparams containing: 
        - out_dir
        - num_train_epochs
        - learning_rate 
        - batch_size 
    '''

    args = TrainingArguments(
        output_dir = train_dir or os.path.join(h.out_dir, 'train', h.name), 

        num_train_epochs = h.num_train_epochs,
        learning_rate = h.learning_rate,
        per_device_train_batch_size = h.batch_size,
        per_device_eval_batch_size = h.batch_size,

        evaluation_strategy='steps',
        eval_steps = 0.25 / h.num_train_epochs, # Every quarter epoch 
        save_strategy='steps',
        save_steps = 0.25 / h.num_train_epochs, # Every quarter epoch

        # weight_decay=0.01,
        load_best_model_at_end=h.load_best_model_at_end,
        metric_for_best_model='eval_macro avg',
        seed=h.seed,
        use_cpu=False,
        local_rank=1,

        logging_first_step = True, 
        report_to = 'wandb',
        logging_steps = 10,
    )

    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = encoded_dataset['train'],
        eval_dataset = encoded_dataset['eval'],
        compute_metrics = compute_metrics
    )

    return trainer

## NOTE: Data

class QueryData(dict):
    def __str__(self):
        return '\n\t' + '\n\t'.join([
            f'{k:16} {len(v["X_train"]):5} {len(v["X_eval"]):5} {len(v["X_test"]):5}'
            for k, v in self.items()]
        )
    
    def __repr__(self):
        return self.__str__()

def get_contextual_query_data(as_query: Query = CopilotQuery, full_reload=False, train_size=0.8, queries=None, debug=False):
    ''' Returns a dictionary of train/eval/test partitions of query data, with the following keys:
        `unbalanced`, `classes`, `subclasses`, `biased`, `oversampled`, `oversampled_biased`
        The train/eval/test splits (even under different distributions) are completely disjoint.
        where
            `unbalanced` is the original data distribution, 
            `classes` undersamples to balance positive/negative classes,
            `subclasses` undersamples to balance manual/auto in positive class,
            `biased` undersamples to balance manual/auto_acc/auto_rej subclasses,
            `oversampled` oversamples to balance positive/negative classes (should actually be `oversampled_classes` to stick with the above nomenclature), 
            `oversampled_biased` oversamples to balance manual/auto_acc/auto_rej subclasses.
            
        Params: 
            as_query: maps to the given Query subclass
            full_reload: if True, repartitions from all D.SORTED_DATA, otherwise uses pre-saved index pickles
    '''
    positive = lambda q: q.is_manually_invoked() or q.is_accepted() 

    if full_reload or queries is not None:

        queries = Queries(as_query=as_query) if queries is None else queries

        # NOTE: Filtering steps: we want unmodified contextual queries with at least 10 chars of context
        queries = [q for q in queries if \
                q.has_context() and not q.has_modified_prefix() and q.get_document_length() >= 10]

        # NOTE: Separate into positive and negative classes 
        positive_queries, negative_queries = [], []
        for query in queries:
            if positive(query): positive_queries.append(query)
            else: negative_queries.append(query)

        # NOTE: Saving indices as pickles for future re-use
        if full_reload: 
            positive_queries = Queries(positive_queries, 'positive_contextual')
            negative_queries = Queries(negative_queries, 'negative_contextual')

            positive_queries.save() 
            negative_queries.save() 

            positive_queries, negative_queries = np.array(positive_queries.queries), np.array(negative_queries.queries)
        else: 
            positive_queries, negative_queries = np.array(positive_queries), np.array(negative_queries)

    else: 
        # positive_queries = np.array([as_query(q) for q in Queries('positive_contextual')])
        # negative_queries = np.array([as_query(q) for q in Queries('negative_contextual')])
        positive_queries = np.array(Queries('positive_contextual', as_query=as_query))
        negative_queries = np.array(Queries('negative_contextual', as_query=as_query))

    # TODO: see whether we can pass down the smaller, filtered query_data every time
    # Simply create train/eval/test splits
    unbalanced = _create_splits(positive_queries, negative_queries, train_size, debug)
    # Modify train/eval/test splits to have balanced 1:1 positive:negative classes
    classes = _apply_to_splits(_undersample_classes, unbalanced)
    # Modify train/eval/test splits to have balanced 1:1 manual:automatic subclasses in positive class
    subclasses = _apply_to_splits(_undersample_subclasses, classes)
    # Modify train/eval/test splits to have balanced 1:1:1 manual:auto_acc:auto_rej subclasses
    biased = _apply_to_splits(_undersample_biased, subclasses)

    # Oversample from unbalanced data to have 1:1 positive:negative classes
    oversampled = _apply_to_splits(_oversample_classes, unbalanced)
    # Oversample from unbalanced data to have 1:1:1 manual:auto_acc:auto_rej subclasses
    oversampled_biased = _apply_to_splits(_oversample_biased, unbalanced)


    query_data = QueryData({
        'unbalanced': unbalanced,
        'classes': classes,
        'subclasses': subclasses,
        'biased': biased,
        'oversampled': oversampled,
        'oversampled_biased': oversampled_biased
    })

    print(f'\t\033[1mdistribution \t\ttrain \teval \ttest \033[0m')
    for partition, dataset in query_data.items():
        print(f'\t{partition:16} \t{len(dataset["X_train"]):5} \t{len(dataset["X_eval"]):5} \t{len(dataset["X_test"]):5}')
    print('\n\n')

    # override the __str__ method of query_data to be less annoying
    return query_data

def _create_splits(positive_queries, negative_queries, train_size, debug=False):
    ''' Split the unbalanced data into train/eval/test splits '''
    # NOTE: 'unbalanced': data following real-life distribution
    X_train_queries, X_other, y_train, y_other = train_test_split(
        np.concatenate((positive_queries, negative_queries)),
        np.concatenate((np.ones(len(positive_queries)), np.zeros(len(negative_queries)))),
        random_state = 42,
        train_size=train_size,
    )

    X_eval_queries, X_test_queries, y_eval, y_test = train_test_split(
        X_other,
        y_other,
        random_state = 42,
        test_size=0.5,
    )

    s = slice(320) if debug else slice(None,None,None)

    return {
        'X_train': X_train_queries[s], 'X_eval': X_eval_queries[s],
        'X_test': X_test_queries[s], 'y_train': y_train[s],
        'y_eval': y_eval[s], 'y_test': y_test[s] 
    }

def _apply_to_splits(fn, query_data, splits=['train', 'eval', 'test']):
    ''' Helper function to ensure exclusivity between splits. 
        Apply a function to each split in query_data, and return new query_data'''

    new_query_data = {} 
    for split in splits:
        X_label, y_label = f'X_{split}', f'y_{split}'
        new_query_data[X_label], new_query_data[y_label] = fn(query_data[X_label], query_data[y_label])
    return new_query_data

def _undersample_classes(X, y):
    ''' Undersample the unbalanced data split to have 1:1 positive:negative classes '''
    positive_queries, negative_queries = [], []
    for query, label in zip(X, y):
        if label == 1: positive_queries.append(query)
        else: negative_queries.append(query)
    
    n_samples = min(len(positive_queries), len(negative_queries))
    sampled_positive_queries = np.array(positive_queries)[np.random.choice(len(positive_queries), n_samples, replace=False)]
    sampled_negative_queries = np.array(negative_queries)[np.random.choice(len(negative_queries), n_samples, replace=False)]

    X_queries = np.concatenate((sampled_positive_queries, sampled_negative_queries))
    y = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))

    # shuffle with np, making sure to keep X and y in sync
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_queries))
    return X_queries[shuffled_indices], y[shuffled_indices]

def _undersample_subclasses(X, y): 
    ''' Undersample from `classes` to ensure equal subclasses (man/auto) in the positive class '''

    queries_dict = { 'man acc': [], 'man rej': [], 'auto acc': [], 'auto rej': [] }
    
    for query in X:
        label = 'man' if query.is_manually_invoked() else 'auto'
        label += ' acc' if query.is_accepted() else ' rej'
        queries_dict[label].append(query)

    # NOTE: What we do in the real world where only 192 manual invocations were accepted:
    n_queries = len(queries_dict['auto acc'])

    manual_accepted = queries_dict['man acc']
    manual_rejected = np.random.choice(queries_dict['man rej'], n_queries - len(manual_accepted), replace=False)
    automatic_accepted = queries_dict['auto acc']
    automatic_rejected = np.random.choice(queries_dict['auto rej'], 2*n_queries, replace=False)

    X_queries = np.concatenate((manual_accepted, manual_rejected, automatic_accepted, automatic_rejected))
    y = np.concatenate((np.ones(n_queries*2), np.zeros(n_queries*2)))

    assert len(X_queries) == len(y), f'Expected {len(X_queries)} queries, got {len(y)}'

    # shuffle with np, making sure to keep X and y in sync
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_queries))
    return X_queries[shuffled_indices], y[shuffled_indices]

def _undersample_biased(X, y):
    ''' undersample classes to balance 1:1:1 manual:auto_acc:auto_rej subclasses '''

    queries_dict = { 'man acc': [], 'man rej': [], 'auto acc': [], 'auto rej': [] }
    
    for query in X:
        label = 'man' if query.is_manually_invoked() else 'auto'
        label += ' acc' if query.is_accepted() else ' rej'
        queries_dict[label].append(query)

    n_queries = len(queries_dict['auto acc'])

    manual_accepted = queries_dict['man acc']
    manual_rejected = np.random.choice(queries_dict['man rej'], n_queries - len(manual_accepted), replace=False)
    automatic_accepted = queries_dict['auto acc']
    automatic_rejected = np.random.choice(queries_dict['auto rej'], n_queries, replace=False)

    X_queries = np.concatenate((manual_accepted, manual_rejected, automatic_accepted, automatic_rejected))
    y = np.concatenate((np.ones(n_queries*2), np.zeros(n_queries)))

    assert len(X_queries) == len(y), f'Expected {len(X_queries)} queries, got {len(y)}'

    # shuffle with np, making sure to keep X and y in sync
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_queries))
    return X_queries[shuffled_indices], y[shuffled_indices]
        
def _oversample_classes(X, y):
    ''' Oversample the unbalanced data to have 1:1 positive:negative classes '''

    positive_queries, negative_queries = [], []
    for query, label in zip(X, y):
        if label == 1: positive_queries.append(query)
        else: negative_queries.append(query)

    n_samples = max(len(positive_queries), len(negative_queries))

    # replicate the smaller list by the integer factor it is smaller,
    # and then randomly sample the remainder
    smaller_list = positive_queries if len(positive_queries) < len(negative_queries) else negative_queries
    larger_list = positive_queries if len(positive_queries) > len(negative_queries) else negative_queries

    X_oversample = np.concatenate((
        larger_list, 
        smaller_list * (n_samples // len(smaller_list)),
        np.random.choice(smaller_list, n_samples % len(smaller_list), replace=False)
    ))
    y_oversample = np.concatenate((np.ones(n_samples), np.zeros(n_samples))) \
        if larger_list is positive_queries else np.concatenate((np.zeros(n_samples), np.ones(n_samples)))
    
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_oversample))
    return X_oversample[shuffled_indices], y_oversample[shuffled_indices]

def _oversample_biased(X, y):
    ''' Oversample the unbalanced data to have 1:1:1 manual:auto_acc:auto_rej subclasses '''

    queries_dict = {'man': [], 'auto acc': [], 'auto rej': []}
    for query in X: 
        if query.is_manually_invoked():
            queries_dict['man'].append(query)
        elif query.is_accepted():
            queries_dict['auto acc'].append(query)
        else:
            queries_dict['auto rej'].append(query)

    # There exist around 260 manual rejected queries for every manual accepted query. 
    man_acc = [q for q in queries_dict['man'] if q.is_accepted()]
    queries_dict['man'] += man_acc * 200

    max_label, max_size = max(queries_dict, key=lambda k: len(queries_dict[k])), max(len(queries_dict[key]) for key in queries_dict)
    assert max_label == 'auto rej', 'I am not going to deal with the other cases' 

    X_oversample = np.concatenate([
        queries_dict['auto rej'],
        queries_dict['auto acc'] * (max_size // len(queries_dict['auto acc'])),
        np.random.choice(queries_dict['auto acc'], max_size % len(queries_dict['auto acc']), replace=False),
        queries_dict['man'] * (max_size // len(queries_dict['man'])),
        np.random.choice(queries_dict['man'], max_size % len(queries_dict['man']), replace=False)
    ])
    y_oversample = np.concatenate((np.zeros(max_size), np.ones(max_size*2)))

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(X_oversample))
    return X_oversample[shuffled_indices], y_oversample[shuffled_indices]

from datasets import DatasetDict, concatenate_datasets
from sklearn.model_selection import StratifiedKFold

def fold_iterator(dataset_dict: DatasetDict | dict, second_dict: dict = None, 
                  train_size = 0.9, seed = 42): 
    ''' Creates an iterator over the 10 folds of a dataset by default. 
        Unfortunately the return type is an absolute mess for backwards compatibility,
        Something like Generator | dict[str, np.ndarray] | tuple of '''

    # why sklearn makes you figure this out on your own is beyond me
    n_splits = int(1/(1-train_size) - 1)  
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if second_dict is not None: 

        combined = concatenate_datasets([dataset_dict['train'], dataset_dict['eval']])
        X, y = combined['input_ids'], combined['label']

        X_2 = np.concatenate([second_dict['X_train'], second_dict['X_eval']])
        y_2 = np.concatenate([second_dict['y_train'], second_dict['y_eval']])

        for train_index, test_index in skf.split(X, y):
            yield combined.select(train_index), combined.select(test_index),\
                    {'X_train': X_2[train_index], 'y_train': y_2[train_index],\
                     'X_eval': X_2[test_index], 'y_eval': y_2[test_index]}
        return 

    elif isinstance(dataset_dict, DatasetDict):
        combined = concatenate_datasets([dataset_dict['train'], dataset_dict['eval']])
        X, y = combined['input_ids'], combined['label']

        for train_index, test_index in skf.split(X, y):
            yield combined.select(train_index), combined.select(test_index)
        return
    else: 
        X = np.concatenate([dataset_dict['X_train'], dataset_dict['X_eval']])
        y = np.concatenate([dataset_dict['y_train'], dataset_dict['y_eval']])

        for train_index, test_index in skf.split(X, y):
            yield {'X_train': X[train_index], 'y_train': y[train_index],\
                   'X_eval': X[test_index], 'y_eval': y[test_index]}


## Debugging 

from transformers.models.roberta import RobertaForSequenceClassification
base_model = RobertaForSequenceClassification.from_pretrained('huggingface/CodeBERTa-small-v1', config='huggingface/CodeBERTa-small-v1')
base_params = {n: p for n, p in base_model.named_parameters()}

def print_new_params(model, freeze_existing=False):

    new_params = {} 
    print('\n\t\033[1mNew Parameters\033[0m')

    for name, param in model.named_parameters():
        if name not in base_params: 
            print(f'    {str(param.shape):20} \t{name:50} \t{param.requires_grad}')
            new_params[name] = param 
        elif freeze_existing: 
            param.requires_grad = False

    new_params = sum(param.numel() for param in new_params.values())
    total_params = sum(param.numel() for param in model.parameters())
    learnable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_params = sum(param.numel() for param in model.parameters() if not param.requires_grad)

    print(f'''
        {new_params:,} new weights included in {total_params:,} total.
        {learnable_params:,} learnable, {frozen_params:,} frozen. 
        ''')

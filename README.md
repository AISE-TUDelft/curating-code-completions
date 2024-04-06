## Curating Code Completions
Replication Package for "A Transformer-Based Approach for Smart Invocation of Automatic Code Completion" @ ACM AIWARE'24. 

The repository is structured as follows: 

[`appendix.pdf`](./appendix.pdf) is the online appendix, containing results from additional exploration and experimentation beyond the scope of the paper. Specifically: 

1. Analysis of the features used in Copilot's Filter. 
2. Comparison of our novel tokenisation strategy against baselines. 
3. The effect of class-distribution on CodeBERTa and Logistic Regression performance. 
4. Performance of JonBERTa-head architecture variations. 
5. Performance of JonBERTa-attn architecture variations. 
6. Exhaustive comparison of alternative logistic-regression approaches. 


##### Offline
Our training & inference scripts are prefixed with a `1`. 
- `xx_[logres | codeberta | jonberta]_classifier.py` are the training scripts for the Logistic Regression, CodeBERTa, and JonBERTa models, respectively. 
- [`modeling_jonberta`](./modeling_jonberta.py) is a Pytorch implementation of JonBERTa. 
- [`14_eval_models.ipynb`](./14_eval_models.ipynb) contains our test setup. 

Our evaluation scripts are prefixed with a `2`. 
- `20_statistics.ipynb` bootstraps the results as described in the Evaluation Metrics sub-section of the Experimental Setup. 
- `21_user_study.ipynb` tracks usage data of the deployed filters. 
- `22_codebertscore.ipynb` performs CodeBERTScore computation on the accepted-completion / ground-truth pairs. 

##### Online
References to online plugin repository will be added after the review process. 


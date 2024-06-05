## Curating Code Completions
Replication Package for [***"A Transformer-Based Approach for Smart Invocation of Automatic Code Completion"***](https://huggingface.co/collections/AISE-TUDelft/smart-invocation-of-code-completion-66473ddf6fa6cf6e541f750c) @ FSE [AIWARE'24](https://2024.aiwareconf.org/track/aiware-2024-papers). 

#### Appendix to the Paper
[`appendix.pdf`](./appendix.pdf) is the online appendix, containing results from additional exploration and experimentation beyond the scope of the paper. It is also available in the full paper on [Arxiv](https://arxiv.org/abs/2405.14753). Specifically: 

1. Analysis of the features used in Copilot's Filter. 
2. Comparison of our novel tokenisation strategy against baselines. 
3. The effect of class-distribution on CodeBERTa and Logistic Regression performance. 
4. Performance of JonBERTa-head architecture variations. 
5. Performance of JonBERTa-attn architecture variations. 
6. Exhaustive comparison of alternative logistic-regression approaches for integrating (tokenised) code context with scalar telemetry features. 


#### Offline Experiments
Our training & inference scripts are prefixed with a `1`. 
- `xx_[logres | codeberta | jonberta]_classifier.py` are the training scripts for the Logistic Regression, CodeBERTa, and JonBERTa models, respectively. 
- [`modeling_jonberta`](./modeling_jonberta.py) is a Pytorch implementation of JonBERTa. 
- [`14_eval_models.ipynb`](./14_eval_models.ipynb) contains our test setup. 

Our evaluation scripts are prefixed with a `2`. 
- `20_statistics.ipynb` bootstraps the results as described in the Evaluation Metrics sub-section of the Experimental Setup. 
- `21_user_study.ipynb` tracks usage data of the deployed filters. 
- `22_codebertscore.ipynb` performs CodeBERTScore computation on the accepted-completion / ground-truth pairs. 

#### Online Experiments
Implementation of the filters for the Code4Me user study can be found in the [code4me public repository](https://github.com/code4me-me/code4me) on GitHub. 

#### Models
The median-performing model across dataset splits, for every hyperparameter combination, is published on our AISE Lab's [huggingface collection](https://huggingface.co/collections/AISE-TUDelft/smart-invocation-of-code-completion-66473ddf6fa6cf6e541f750c).

#### Cite
To cite the paper, you may use 

```bibtex
@misc{de_moor_smart_invocation_2024,
	title = {A {Transformer}-{Based} {Approach} for {Smart} {Invocation} of {Automatic} {Code} {Completion}},
	url = {http://arxiv.org/abs/2405.14753},
	doi = {10.1145/3664646.3664760},
	author = {de Moor, Aral and van Deursen, Arie and Izadi, Maliheh},
	month = may,
	year = {2024},
}
```

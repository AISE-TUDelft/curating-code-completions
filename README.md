<h2 align="center"><b><h3>A Transformer-Based Approach for <br> Smart Invocation of Automatic Code Completion</h3></b></h2><br>


<p align="center">
  <b>Aral de Moor, Arie van Deursen, and Maliheh Izadi</b>
</p>

<p align="center">
  <i>
    Delft University of Technology<br>
    AISE Lab @ Software Engineering Research Group<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://arxiv.org/abs/2405.14753"><b>Full Paper (with Appendices)</b></a><br>
  <a href="https://doi.org/10.1145/3664646.3664760"><b>Workshop Paper</a> (<a href="https://2024.aiwareconf.org/track/aiware-2024-papers">AIWARE</a> @ FSE'24)</b><br>
  <a href="https://huggingface.co/collections/AISE-TUDelft/smart-invocation-of-code-completion-66473ddf6fa6cf6e541f750c"><b>HuggingFace Model Collection</b></a>
</p>
<br>

---
<br>
<h3 align="center"><b>Abstract</b></h3><br>
Transformer-based language models are highly effective for code completion, with much research dedicated to enhancing the content of these completions. Despite their effectiveness, these models come with high operational costs and can be intrusive, especially when they suggest too often and interrupt developers who are concentrating on their work. Current research largely overlooks how these models interact with developers in practice and neglects to address when a developer should receive completion suggestions. To tackle this issue, we developed a machine learning model that can accurately predict when to invoke a code completion tool given the code context and available telemetry data.
<br></br>
To do so, we collect a dataset of 200k developer interactions with our cross-IDE code completion plugin and train several invocation filtering models. Our results indicate that our small-scale transformer model significantly outperforms the baseline while maintaining low enough latency. We further explore the search space for integrating additional telemetry data into a pre-trained transformer directly and obtain promising results. To further demonstrate our approach's practical potential, we deployed the model in an online environment with 34 developers and provided real-world insights based on 74k actual invocations.
</br>

---
<br>

### Contents

This repository contains the (online) appendix, source code used to train our models, and the offline evaluation. 

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

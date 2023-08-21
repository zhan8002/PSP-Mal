# PSP-Mal
> PSP-Mal: Evading Malware Detection via Prioritized Experience-based Reinforcement Learning with Shapley Prior

## Environment
PSP-Mal exposes OpenAI's `gym` environments and extends the [malware_rl](https://github.com/bfilar/malware_rl) repo to allow researchers to develop Reinforcement Learning agents to bypass LightGBM (Light Gradient Boosting Machine) Malware detectors. 
The LightGBM models, respectively trained on [Ember](https://github.com/endgameinc/ember) (Endgame Malware BEnchmark for Research) ([paper](https://arxiv.org/abs/1804.04637)) and [SOREL-20M](https://github.com/sophos/SOREL-20M) (SOREL-20M: A Large Scale Benchmark Dataset for Malicious PE Detection) ([paper](https://arxiv.org/abs/2012.07634)), were implemented to facilitate the comparison of the PSP-Mal and baseline methods.

### Action Space
Actions include a variety of non-breaking (e.g. binaries will still execute) modifications to the PE header, sections, imports and overlay and are listed below.
![image](actionset.jpg)

### Observation Space
The `observation_space` of the `gym` environments are a feature vector representing the malware sample. In this work， `numpy.array == 2381`.

### Agents
To guide the agent training, we use the [SHAP](https://github.com/shap/shap) to explain the black-box model and exploit information as a Shapley prior. Then, two modules are proposed to further improve RL: Shapley prior-based prioritized experience replay, and Thompson sampling within actions. Finally, a D3QN RL algorithm is instantiated using the above modules.


## Setup
The PSP-Mal framework is built on Python3.7 we recommend first creating a virtualenv  then performing the following actions ensure you have the correct python libraries:

```sh
pip install -r requirements.txt
```

PSP-Mal leverages the [LIEF](https://github.com/lief-project/LIEF) and [PEfile](https://github.com/erocarrera/pefile) to perform functionality-preserving modifications.

To get `PSP-Mal` running you need prepare the following data:

- Target detector - [Ember](https://github.com/Azure/2020-machine-learning-security-evasion-competition/blob/master/defender/defender/models/ember_model.txt.gz),  [SOREL-20M](https://github.com/sophos-ai/SOREL-20M) models. Model files need to be placed into the `malware_rl/envs/utils/` directory.

- Benign binaries - a small set of "trusted" binaries (e.g. grabbed from base Windows installation) you can download some via [PortableApps](https://portableapps.com). And then store these binaries in `malware_rl/envs/controls/trusted`

- Malware samples - Though we cannot release the malware downloaded by [Virustotal](https://www.virustotal.com/)，we offer the list of hashes of samples in `sample_hash.txt`


## Baselines
![image](baselines.jpg)


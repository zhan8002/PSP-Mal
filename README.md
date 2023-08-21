# PSP-Mal
> PSP-Mal: Evading Malware Detection via Prioritized Experience-based Reinforcement Learning with Shapley Prior
> 
## Background
The manipulation environment of PSP-Mal uses OpenAI's gym environments. 


## Environment
MalwareRL exposes `gym` environments for Ember and MalConv to allow researchers to develop Reinforcement Learning agents to bypass Malware Classifiers. 
The LightGBM (Light Gradient Boosting Machine) models, respectively trained on [Ember](https://github.com/endgameinc/ember) (Endgame Malware BEnchmark for Research) ([paper](https://arxiv.org/abs/1804.04637)) and [SOREL-20M](https://github.com/sophos/SOREL-20M) (SOREL-20M: A Large Scale Benchmark Dataset for Malicious PE Detection) ([paper](https://arxiv.org/abs/2012.07634)), were implemented to facilitate the comparison of the PSP-Mal and baseline methods.
### Action Space
Actions include a variety of non-breaking (e.g. binaries will still execute) modifications to the PE header, sections, imports and overlay and are listed below.
![image](actionset.PNG)

### Observation Space
The `observation_space` of the `gym` environments are an array representing the feature vector. For ember this is `numpy.array == 2381` and malconv `numpy.array == 1024**2`. The MalConv gym presents an opportunity to try RL techniques to generalize learning across large State Spaces.

### Agents
A baseline agent `RandomAgent` is provided to demonstrate how to interact w/ `gym` environments and expected output. This agent attempts to evade the classifier by randomly selecting an action. This process is repeated up to the length of a game (e.g. 50 mods). If the modifed binary scores below the classifier threshold we register it as an evasion. In a lot of ways the `RandomAgent` acts as a fuzzer trying a bunch of actions with no regard to minimizing the modifications of the resulting binary.

Additional agents will be developed and made available (both model and code) in the coming weeks.

**Table 1:** _Evasion Rate against Ember Holdout Dataset_*
| gym | agent | evasion_rate | avg_ep_len |
| --- | ----- | ------------ | ---------- |
| ember | RandomAgent | 89.2% | 8.2
| malconv | RandomAgent | 88.5% | 16.33



## Setup
To get `malware_rl` up and running you will need the follow external dependencies:
- [LIEF](https://lief.quarkslab.com/)
- [Ember](https://github.com/Azure/2020-machine-learning-security-evasion-competition/blob/master/defender/defender/models/ember_model.txt.gz), [Malconv](https://github.com/endgameinc/ember/blob/master/malconv/malconv.h5) and [SOREL-20M](https://github.com/sophos-ai/SOREL-20M) models. All of these then need to be placed into the `malware_rl/envs/utils/` directory.
  > The SOREL-20M model requires use of the `aws-cli` in order to get. When accessing the AWS S3 bucket, look in the `sorel-20m-model/checkpoints/lightGBM` folder and fish out any of the models in the `seed` folders. The model file will need to be renamed to `sorel.model` and placed into `malware_rl/envs/utils` alongside the other models.
- UPX has been added to support pack/unpack modifications. Download the binary [here](https://upx.github.io/) and place in the `malware_rl/envs/controls` directory.
- Benign binaries - a small set of "trusted" binaries (e.g. grabbed from base Windows installation) you can download some via MSFT website ([example](https://download.microsoft.com/download/a/c/1/ac1ac039-088b-4024-833e-28f61e01f102/NETFX1.1_bootstrapper.exe)). Store these binaries in `malware_rl/envs/controls/trusted`
- Run `strings` command on those binaries and save the output as `.txt` files in `malware_rl/envs/controls/good_strings`
- Download a set of malware from VirusShare or VirusTotal. I just used a list of hashes from the Ember dataset

I used a [conda](https://docs.conda.io/en/latest/) env set for Python3.7:

`conda create -n malware_rl python=3.7`

Finally install the Python3 dependencies in the `requirements.txt`.

`pip3 install -r requirements.txt`


### Papers

- Song, Wei, et al. "Automatic Generation of Adversarial Examples for Interpreting Malware Classifiers." arXiv preprint arXiv:2003.03100 (2020).
 ([paper](https://arxiv.org/abs/2003.03100))
- Fang, Zhiyang, et al. "Evading anti-malware engines with deep reinforcement learning." IEEE Access 7 (2019): 48867-48879. ([paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8676031))

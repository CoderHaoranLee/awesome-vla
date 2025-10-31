# Awesome VLA Papers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of papers and resources on Vision-Language-Action (VLA) models in robotics.  

---

## Contents
- [Survey Papers](#survey-papers)
- [VLA Models](#vla-models)
- [Datasets](#datasets)
- [Benchmarks](#benchmarks)
- [Applications](#applications)
- [Related Resources](#related-resources)

---

## Survey Papers
- **A Survey on Vision-Language-Action Models for Embodied AI**,  Arxiv 2024, [[paper](https://arxiv.org/abs/2405.14093)] [[code](https://github.com/yueen-ma/Awesome-VLA)]
- **A Survey on Vision-Language-Action Models: An Action Tokenization Perspective**, Arxiv 2025, [[paper](https://arxiv.org/abs/2507.01925)]
- **Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends**, Arxiv 2025, [[paper](https://arxiv.org/abs/2506.20966)]
- **Vision-Language-Action Models: Concepts, Progress, Applications and Challenges**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.04769)]
- **Vision Language Action Models in Robotic Manipulation: A Systematic Review**, Arxiv 2025, [[paper](https://arxiv.org/abs/2507.10672)]
- **Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey**, Arxiv 2025, [[paper](https://arxiv.org/abs/2508.13073)] [[code](https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation)]

<!-- - **VLA: The Future of Embodied AI** (2024) [[paper](link)] [[code](link)] -->

***

## World Models

- **Ctrl-World: A Controllable Generative World Model for Robot Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.20813)] [[web](https://3dgsworld.github.io/)] [[速读](./papers/world-model/gs-world.md)]
- **OmniNWM: Omniscient Driving Navigation World Models**, CoRL 2025, [[paper](https://www.arxiv.org/abs/2510.18313)] [[web](https://arlo0o.github.io/OmniNWM/)] [[code](https://github.com/Ma-Zhuang/OmniNWM)] [[速读](./papers/world-model/omninwm.md)]
- **Training Agents Inside of Scalable World Models**, Arxiv 2025, [[paper](https://arxiv.org/abs/2509.24527)] [[web](https://danijar.com/project/dreamer4/)]
- **Generative World Modelling for Humanoids1X World Model Challenge Technical Report - Team Revontuli**, Arxiv 2025, [[paper](https://arxiv.org/pdf/2510.07092)]
- **World-in-World: World Models in a Closed-Loop World**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.18135)]
- **GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.20813)] [[web](https://3dgsworld.github.io/)]

***

## VLA Models
### Representation
#### Color Image
- **OpenVLA** (Google DeepMind, 2024) [[paper](link)] [[code](link)]
- **$\pi_0$** ($\pi_0$, 2024) [[paper](link)] [[code](link)]

#### Depth Image
- **3D-VLA: A 3D Vision-Language-Action Generative World Model**, ICML 2024, [[paper](https://arxiv.org/abs/2403.09631)] [[web](https://vis-www.cs.umass.edu/3dvla)] [[code](https://github.com/UMass-Embodied-AGI/3D-VLA)]

#### Point Cloud
- **GeoVLA: Empowering 3D Representations in Vision-Language-Action Models**, Arxiv 2025, [[paper](https://arxiv.org/abs/2508.09071)] [[web](https://linsun449.github.io/GeoVLA/)]

#### Tactile
- **VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.09577v1)] [[web](https://sites.google.com/view/vtla)]
- **Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization**, Arxiv 2025, [[paper](https://arxiv.org/abs/2507.09160)]
- **OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing**, Arxiv 2025, [[paper](https://arxiv.org/abs/2508.08706)]
- **MLA: A Multisensory Language-Action Model for Multimodal Understanding and Forecasting in Robotic Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/pdf/2509.26642)]



#### Force
- **ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.22159)]

***

## Datasets
- **LeRobot** (Meta, 2024) [[paper](link)] [[dataset](link)]

***

## Pre-training

### Phased training with cross domain data

- **X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.10274)] [[web](https://thu-air-dream.github.io/X-VLA/)] [[code](https://github.com/2toinf/X-VLA)] [[速读](./papers/pre-training/phased-training/x-vla.md)]

### Reasoning

- **Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse**, Arxiv 2025, [[paper](https://arxiv.org/abs/2506.07639)]
- **Action-Free Reasoning for Policy Generalization**, Arxiv 2025, [[paper](https://arxiv.org/abs/2502.03729)] [[web](https://rad-generalization.github.io/)] 
- **ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning**, Arxiv 2025, [[paper](https://arxiv.org/abs/2507.16815)] [[web](https://jasper0314-huang.github.io/thinkact-vla/)]
- **Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics**, Arxiv 2025, [[paper](https://arxiv.org/abs/2506.00070)]
- **dVLA: Diffusion Vision-Language-Action Model with Multimodal Chain-of-Thought**, Arxiv 2025, [[paper](https://arxiv.org/abs/2509.25681)]

***

## Memory

**MemER: Scaling Up Memory for Robot Control via Experience Retrieval**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.20328)] [[web](https://jen-pan.github.io/memer/)] [[速读](./papers/memory/memer.md)]

***
## Post-training
### SFT

---
### RL

#### Real-world
- **Improving Vision-Language-Action Model with Online Reinforcement Learning**, ICRA 2025, [[paper](https://arxiv.org/abs/2501.16664)]
- **VLAC: A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning**, Arxiv 2025, [[paper](https://arxiv.org/abs/2509.15937)] [[code](https://github.com/InternRobotics/VLAC)] [[速读](./papers/post-training/RFT/real-world/vlac.md)]
- **RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.14830)] [[web](https://lei-kun.github.io/RL-100/)] 
- **Self-Improving Embodied Foundation Models**, Arxiv 2025, [[paper](https://arxiv.org/pdf/2509.15155v1)]
- **Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Class and Backbone**, Arxiv 2025, [[paper](https://arxiv.org/pdf/2412.06685)]
- **RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning**, Arxiv 2024, [[paper](https://arxiv.org/abs/2412.09858)]
- **ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning**, Arxiv 2025, [[paper](https://arxiv.org/pdf/2505.07395)]
- **ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations**, CoRL 2025, [[paper](https://arxiv.org/pdf/2505.10911)] [[web](https://rewind-reward.github.io/)] [[code](https://github.com/rewind-reward/ReWiND)]
- **Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model**, ICLR 2025, [[paper](https://arxiv.org/abs/2412.13630)] [[code](https://github.com/tongzhoumu/policy_decorator)]
- **ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy**, RSS 2025, [[paper](https://arxiv.org/abs/2502.05450)] [[web](https://cccedric.github.io/conrft/)] [[code](https://github.com/cccedric/conrft)] [[速读](https://www.jiqizhixin.com/articles/2025-04-18?from=synced&keyword=conrft)]
- **RM-RL: Role-Model Reinforcement Learning for Precise Robot Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.15189)]
- **SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2509.25358)] [[web](https://qianzhong-chen.github.io/sarm.github.io/)]
- **Self-Improving Vision-Language-Action Models with Data Generation via Residual RL**, Underreview ICLR 2026, [[paper](https://openreview.net/forum?id=eUGoqrZ6Ea)]


#### Simulator
- **GRAPE: Generalizing Robot Policy via Preference Alignment**, Arxiv 2025, [[paper](https://arxiv.org/abs/2411.19309)] [[web](https://grape-vla.github.io/)] [[code](https://github.com/aiming-lab/grape)]
- **What Can RL Bring to VLA Generalization? An Empirical Study**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.19789)] [[web](https://rlvla.github.io/)] [[code](https://github.com/gen-robot/RL4VLA)]
- **SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning**, Arxiv 2025, [[paper](https://arxiv.org/abs/2509.09674)] [[code](https://github.com/PRIME-RL/SimpleVLA-RL)]
- **VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.18719)] [[code](https://github.com/GuanxingLu/vlarl)]
- **TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization**, Arxiv 2025, [[paper](https://arxiv.org/abs/2506.08440)]
- **RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.06710)]
- **Interactive Post-Training for Vision-Language-Action Models**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.17016)] [[code](https://github.com/Ariostgx/ript-vla)]
- **PROGRESSIVE STAGE-AWARE REINFORCEMENT FORFINE-TUNING VISION-LANGUAGE-ACTION MODELS**, Underreview ICLR 2026, [[paper](https://openreview.net/attachment?id=qBcgyxDeMM&name=pdf)]
- **Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models**, Arxiv 2025, [[paper](https://arxiv.org/pdf/2510.09976)]
- **ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning**, Arxiv 2025, [[paper](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/09dbaac9-e1ab-4e18-abf2-99ec82476290.pdf)]
- **π_\texttt{RL}: Online RL Fine-tuning for Flow-based Vision-Language-Action Models***, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.25889)] [[code](https://github.com/RLinf/RLinf)]


#### World Model
- **VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators**， Arxiv 2025, [[paper](https://arxiv.org/abs/2510.00406)] [[web](https://vla-rft.github.io/)] [[code](https://github.com/OpenHelix-Team/VLA-RFT)] [[速读](./papers/post-training/RFT/simulator/vla-rft.md)]
- **Ctrl-World: A Controllable Generative World Model for Robot Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.10125)] [[code](https://github.com/Robert-gyj/Ctrl-World)] [[速读](./papers/post-training/RFT/world-model/ctrl-world.md)]
- **World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2509.19080)] [[web](https://world4rl.github.io/)]

---

### Test-Time Scaling
- **Verifier-free Test-Time Sampling for Vision Language Action Models**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.05681)] [[速读](./papers/post-training/TTS/mg-select.md)]
- **RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models**, Arxiv 2025, [[paper](https://arxiv.org/abs/2506.17811)]
- **Hume: Introducing System-2 Thinking in Visual-Language-Action Model**, Arxiv 2025, [[paper](https://arxiv.org/pdf/2505.21432)] [[web](https://hume-vla.github.io/)] [[code](https://github.com/hume-vla/hume)]
- **From Foresight to Forethought: VLM-In-the-Loop Policy Steering via Latent Alignment**, Arxiv 2025, [[paper](https://arxiv.org/abs/2502.01828)] [[web](https://yilin-wu98.github.io/forewarn/)]
- **Inference-Time Policy Steering through Human Interactions**, Arxiv 2025, [[paper](https://arxiv.org/abs/2411.16627)] [[web](https://yanweiw.github.io/itps/)]
- **Improving Robotic Foundation Models via Value Guidance**, CoRL 2024, [[paper](https://arxiv.org/abs/2410.13816)] [[web](https://nakamotoo.github.io/V-GPS/)] [[code](https://github.com/nakamotoo/V-GPS)]
- **Learning Affordances at Inference-Time for Vision-Language-Action Models**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.19752)]
- **VLA-Reasoner: Empowering Vision-Language-Action Models with Reasoning via Online Monte Carlo Tree Search**, Arxiv 2025, [[paper](https://arxiv.org/pdf/2509.22643)]

***
## Evaluation

### Real-World

### Simulator

### World Model
- **1X World Model: Evaluating Bits, not Atoms**, 2025, [[paper](https://www.1x.tech/1x-world-model.pdf)] [[code](https://github.com/1x-technologies/1xgpt)]
- **Vid2World: crafting video diffusion models to interactive world models**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.14357)]
- **WorldEval: World Model as Real-World Robot Policies Evaluator**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.19017)] [[web](https://worldeval.github.io/)] [[code](https://github.com/liyaxuanliyaxuan/Worldeval)]
---

## Contributing
Contributions are welcome! Please submit a pull request.

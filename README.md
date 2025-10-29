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

## World Models

- **Ctrl-World: A Controllable Generative World Model for Robot Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.20813)] [[web](https://3dgsworld.github.io/)] [[速读](./papers/world-model/gs-world.md)]
- **OmniNWM: Omniscient Driving Navigation World Models**, CoRL 2025, [[paper](https://www.arxiv.org/abs/2510.18313)] [[web](https://arlo0o.github.io/OmniNWM/)] [[code](https://github.com/Ma-Zhuang/OmniNWM)] [[速读](./papers/world-model/omninwm.md)]

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

#### Force
- **ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.22159)]


## Datasets
- **LeRobot** (Meta, 2024) [[paper](link)] [[dataset](link)]


## Pre-training

### Phased training with cross domain data

- **X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.10274)] [[web](https://thu-air-dream.github.io/X-VLA/)] [[code](https://github.com/2toinf/X-VLA)] [[速读](./papers/pre-training/phased-training/x-vla.md)]

### Reasoning

- **Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse**, Arxiv 2025, [[paper](https://arxiv.org/abs/2506.07639)]
- **Action-Free Reasoning for Policy Generalization**, Arxiv 2025, [[paper](https://arxiv.org/abs/2502.03729)] [[web](https://rad-generalization.github.io/)] 
- **ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning**, Arxiv 2025, [[paper](https://arxiv.org/abs/2507.16815)] [[web](https://jasper0314-huang.github.io/thinkact-vla/)]
- **Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics**, Arxiv 2025, [[paper](https://arxiv.org/abs/2506.00070)]



## Memory

**MemER: Scaling Up Memory for Robot Control via Experience Retrieval**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.20328)] [[web](https://jen-pan.github.io/memer/)] [[速读](./papers/memory/memer.md)]

## Post-training
### SFT

### RL

#### Real-world
- **Improving Vision-Language-Action Model with Online Reinforcement Learning**, ICRA 2025, [[paper](https://arxiv.org/abs/2501.16664)]
#### Simulator
- **GRAPE: Generalizing Robot Policy via Preference Alignment**, Arxiv 2025, [[paper](https://arxiv.org/abs/2411.19309)] [[web](https://grape-vla.github.io/)] [[code](https://github.com/aiming-lab/grape)]
- **What Can RL Bring to VLA Generalization? An Empirical Study**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.19789)] [[web](https://rlvla.github.io/)] [[code](https://github.com/gen-robot/RL4VLA)]
- **VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators**， Arxiv 2025, [[paper](https://arxiv.org/abs/2510.00406)] [[web](https://vla-rft.github.io/)] [[code](https://github.com/OpenHelix-Team/VLA-RFT)] [[速读](./papers/post-training/RFT/simulator/vla-rft.md)]
- **SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning**, Arxiv 2025, [[paper](https://arxiv.org/abs/2509.09674)] [[code](https://github.com/PRIME-RL/SimpleVLA-RL)]
- **VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning**, Arxiv 2025, [[paper](https://arxiv.org/abs/2505.18719)] [[code](https://github.com/GuanxingLu/vlarl)]
- **TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization**, Arxiv 2025, [[paper](https://arxiv.org/abs/2506.08440)]
- **RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.06710)]
#### World Model

- **Ctrl-World: A Controllable Generative World Model for Robot Manipulation**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.10125)] [[code](https://github.com/Robert-gyj/Ctrl-World)] [[速读](./papers/post-training/RFT/world-model/ctrl-world.md)]

### Test-Time Scaling
- **Verifier-free Test-Time Sampling for Vision Language Action Models**, Arxiv 2025, [[paper](https://arxiv.org/abs/2510.05681)] [[速读](./papers/post-training/TTS/mg-select.md)]
## Evaluation
- **ALOHA Benchmark** (2023) [[paper](link)] [[website](link)]

---

## Contributing
Contributions are welcome! Please submit a pull request.

# QuLab Infinite - Algorithm & Technique Research 2024-2025

This document summarizes recent research papers that are relevant to the QuLab Infinite project. The papers have been selected to highlight potential new algorithms, techniques, and improvements that could be integrated into the simulation platform.

## 1. Variational Quantum Eigensolvers (VQE) for Molecular Energies

Research in VQE is crucial for enhancing the `quantum_lab`'s capabilities in quantum chemistry simulations. These papers explore improvements in efficiency, noise mitigation, and novel approaches to the VQE algorithm.

---

### SMILES-Inspired Transfer Learning for Quantum Operators in Generative Quantum Eigensolver
- **Authors**: Zhi Yin, Xiaoran Li, Shengyu Zhang, Xin Li, Xiaojin Zhang
- **Published**: 2025-09-24T02:54:09Z
- **Link**: [http://arxiv.org/abs/2509.19715v1](http://arxiv.org/abs/2509.19715v1)
- **Summary**: This paper proposes a transfer learning framework for Generative Quantum Eigensolvers (GQE) inspired by SMILES representations in chemistry. By representing UCCSD quantum operators as text, the method leverages similarities between molecular systems to reduce computational costs for ground-state energy calculations. This could be a valuable technique for accelerating simulations in the `chemistry_lab` by reusing knowledge from previous calculations.

---

### A Full Quantum Eigensolver for Quantum Chemistry Simulations
- **Authors**: Shijie Wei, Hang Li, GuiLu Long
- **Published**: 2019-08-21T15:32:48Z
- **Link**: [http://arxiv.org/abs/1908.07927v2](http://arxiv.org/abs/1908.07927v2)
- **Summary**: The authors propose a "Full Quantum Eigensolver" (FQE) that uses quantum gradient descent, removing the need for a classical optimizer as in traditional VQE. This approach could lead to faster convergence and could be implemented on near-term quantum computers. This represents a potential long-term evolution for the VQE algorithms in `quantum_lab`.

---

### Noise-Mitigated Variational Quantum Eigensolver with Pre-training and Zero-Noise Extrapolation
- **Authors**: Wanqi Sun, Jungang Xu, Chenghua Duan
- **Published**: 2025-01-03T05:34:36Z
- **Link**: [http://arxiv.org/abs/2501.01646v2](http://arxiv.org/abs/2501.01646v2)
- **Summary**: This work focuses on mitigating noise in VQE calculations on current quantum hardware. It combines pre-training of quantum circuits (inspired by matrix product states) with zero-noise extrapolation to improve accuracy. The intelligent grouping of Hamiltonian measurements could also be a practical improvement for our `quantum_lab` simulations.

---

### A blueprint for a Digital-Analog Variational Quantum Eigensolver using Rydberg atom arrays
- **Authors**: Antoine Michel, Sebastian Grijalva, Loïc Henriet, Christophe Domain, Antoine Browaeys
- **Published**: 2023-01-16T14:56:58Z
- **Link**: [http://arxiv.org/abs/2301.06453v2](http://arxiv.org/abs/2301.06453v2)
- **Summary**: This paper explores a digital-analog approach to VQE using Rydberg atom arrays. While QuLab Infinite is a software simulator, the concepts from this hardware-focused paper could inspire new variational ansatzes or simulation methodologies for specific molecular Hamiltonians.

---

### $O(N^3)$ Measurement Cost for Variational Quantum Eigensolver on Molecular Hamiltonians
- **Authors**: Pranav Gokhale, Frederic T. Chong
- **Published**: 2019-08-30T17:36:50Z
- **Link**: [http://arxiv.org/abs/1908.11857v1](http://arxiv.org/abs/1908.11857v1)
- **Summary**: This paper addresses the measurement overhead in VQE. The authors provide a method to partition Hamiltonian terms into commuting families, reducing the measurement scaling from $O(N^4)$ to $O(N^3)$. Implementing this optimization could significantly speed up VQE calculations in the `quantum_lab`.

---

## 2. Machine Learning Force Fields for Molecular Dynamics

Improving the accuracy and efficiency of molecular dynamics is a key goal for the `chemistry_lab`. Machine learning-based force fields offer a promising way to achieve this, moving beyond traditional empirical force fields.

---

### Efficient Machine Learning Force Field for Large-Scale Molecular Simulations of Organic Systems
- **Authors**: Junbao Hu, Liyang Zhou, Jian Jiang
- **Published**: 2023-12-15T02:06:51Z
- **Link**: [http://arxiv.org/abs/2312.09490v1](http://arxiv.org/abs/2312.09490v1)
- **Summary**: The authors propose a universal multiscale higher-order equivariant model for generating ML force fields for large organic systems. Their approach claims high predictive accuracy, improved computational speed, and memory efficiency, which are all critical for the large-scale simulations planned for QuLab Infinite.

---

### Operator Forces For Coarse-Grained Molecular Dynamics
- **Authors**: Leon Klein, Atharva Kelkar, Aleksander Durumeric, Yaoyi Chen, Frank Noé
- **Published**: 2025-06-24T13:51:20Z
- **Link**: [http://arxiv.org/abs/2506.19628v1](http://arxiv.org/abs/2506.19628v1)
- **Summary**: This paper introduces a method for creating coarse-grained force fields using normalizing flows, which reduces the need for extensive atomistic force data. This could be particularly useful for developing new coarse-grained models in the `chemistry_lab` when detailed force information is not available.

---

### Statistically Optimal Force Aggregation for Coarse-Graining Molecular Dynamics
- **Authors**: Andreas Krämer, Aleksander P. Durumeric, Nicholas E. Charron, Yaoyi Chen, Cecilia Clementi, Frank Noé
- **Published**: 2023-02-14T14:35:39Z
- **Link**: [http://arxiv.org/abs/2302.07071v1](http://arxiv.org/abs/2302.07071v1)
- **Summary**: This work focuses on improving the mapping of forces from all-atom to coarse-grained representations. By using an optimized force mapping, more accurate coarse-grained force fields can be learned from the same simulation data. This could lead to more robust and accurate coarse-grained simulations in our system.

---

### Two for One: Diffusion Models and Force Fields for Coarse-Grained Molecular Dynamics
- **Authors**: Marloes Arts, Victor Garcia Satorras, Chin-Wei Huang, Daniel Zuegner, Marco Federici, Cecilia Clementi, Frank Noé, Robert Pinsler, Rianne van den Berg
- **Published**: 2023-02-01T17:09:46Z
- **Link**: [http://arxiv.org/abs/2302.00600v3](http://arxiv.org/abs/2302.00600v3)
- **Summary**: This paper presents a novel approach that uses diffusion generative models to learn a coarse-grained force field without requiring force inputs. The score function of the diffusion model approximates the force field. This is a very interesting approach that could simplify the development of new force fields in the `chemistry_lab`.

---

### Symmetry-adapted graph neural networks for constructing molecular dynamics force fields
- **Authors**: Zun Wang, Chong Wang, Sibo Zhao, Shiqiao Du, Yong Xu, Bing-Lin Gu, Wenhui Duan
- **Published**: 2021-01-08T09:32:24Z
- **Link**: [http://arxiv.org/abs/2101.02930v1](http://arxiv.org/abs/2101.02930v1)
- **Summary**: The authors develop a symmetry-adapted graph neural network (MDGNN) to construct force fields for both molecules and crystals. The architecture preserves physical invariances (translation, rotation, permutation). This is a powerful and general approach that could be used to create high-accuracy force fields for a wide range of systems in QuLab Infinite.

---

## 3. AI for Materials Discovery

The `materials_lab` and the `hive_mind` can benefit greatly from AI-driven approaches to discover new materials. These papers cover generative models, LLMs, and autonomous experimentation systems.

---

### Artificial Intelligence and Generative Models for Materials Discovery -- A Review
- **Authors**: Albertus Denny Handoko, Riko I Made
- **Published**: 2025-08-05T09:56:27Z
- **Link**: [http://arxiv.org/abs/2508.03278v1](http://arxiv.org/abs/2508.03278v1)
- **Summary**: This review provides a broad overview of AI-driven generative models for materials discovery. It discusses different material representations and applications in designing catalysts, semiconductors, polymers, and crystals. This is a good starting point for identifying promising generative models to integrate with the `material_designer.py` component.

---

### Expert-Guided LLM Reasoning for Battery Discovery: From AI-Driven Hypothesis to Synthesis and Characterization
- **Authors**: Shengchao Liu, Hannan Xu, Yan Ai, Huanxin Li, Yoshua Bengio, Harry Guo
- **Published**: 2025-07-21T23:46:11Z
- **Link**: [http://arxiv.org/abs/2507.16110v1](http://arxiv.org/abs/2507.16110v1)
- **Summary**: This paper introduces "ChatBattery," an agentic framework that uses LLMs with domain knowledge to reason about materials design. They successfully used it to discover and synthesize new battery cathode materials. This approach aligns perfectly with the vision for the `hive_mind` and `crystalline_intent` components, suggesting how we could use LLMs for experiment planning.

---

### Adaptive AI decision interface for autonomous electronic material discovery
- **Authors**: Yahao Dai, et al.
- **Published**: 2025-04-17T21:26:48Z
- **Link**: [http://arxiv.org/abs/2504.13344v1](http://arxiv.org/abs/2504.13344v1)
- **Summary**: The authors developed an AI/AE (Autonomous Experimentation) system with an adaptive AI decision interface for discovering electronic materials. The system combines real-time monitoring, data analysis, and human-AI collaboration. This provides a practical blueprint for how the `hive_mind` could orchestrate experiments and adapt its strategy based on incoming results.

---

### AIMS-EREA -- A framework for AI-accelerated Innovation of Materials for Sustainability -- for Environmental Remediation and Energy Applications
- **Authors**: Sudarson Roy Pratihar, Deepesh Pai, Manaswita Nag
- **Published**: 2023-11-18T12:35:45Z
- **Link**: [http://arxiv.org/abs/2311.11060v1](http://arxiv.org/abs/2311.11060v1)
- **Summary**: This paper proposes a framework (AIMS-EREA) that blends materials science theory with generative AI to accelerate the discovery of materials for sustainability. It uses both predictive and generative AI on chemical databases and published research. This aligns with our goal of using the `semantic_lattice` to represent knowledge and drive discovery.

---

### Materials science and engineering: New vision in the era of artificial intelligence
- **Authors**: Tao Qiang, Honghong Gao
- **Published**: 2018-04-23T09:01:57Z
- **Link**: [http://arxiv.org/abs/1804.08293v1](http://arxiv.org/abs/1804.08293v1)
- **Summary**: This paper provides a high-level vision for data-intensive materials science and engineering. It proposes a "DIMSE" (Data-Intensive Materials Science and Engineering) model. While less technical, it reinforces the overall direction of QuLab Infinite and the importance of a data-centric approach.

---

## 4. Automated Synthesis Planning

Automated synthesis planning is a key feature of the `chemistry_lab`. These papers discuss extracting synthesis procedures from text and using AI for planning.

---

### The Materials Science Procedural Text Corpus: Annotating Materials Synthesis Procedures with Shallow Semantic Structures
- **Authors**: Sheshera Mysore, et al.
- **Published**: 2019-05-16T17:57:35Z
- **Link**: [http://arxiv.org/abs/1905.06939v2](http://arxiv.org/abs/1905.06939v2)
- **Summary**: This paper introduces a corpus of materials synthesis procedures annotated with semantic structures. While we are not building an NLP model from scratch, this work highlights the types of structured data that are valuable for synthesis planning and could inform the design of the data structures used in our `synthesis_planner.py`.

---

### MatPROV: A Provenance Graph Dataset of Material Synthesis Extracted from Scientific Literature
- **Authors**: Hirofumi Tsuruta, Masaya Kumagai
- **Published**: 2025-09-01T00:47:27Z
- **Link**: [http://arxiv.org/abs/2509.01042v3](http://arxiv.org/abs/2509.01042v3)
- **Summary**: The authors present MatPROV, a dataset of synthesis procedures represented as provenance graphs. This graph-based representation captures the complex relationships in synthesis procedures. This is a very powerful idea that could be integrated into our `synthesis_planner` and `semantic_lattice` to create more sophisticated synthesis plans.

---

### Multi-tier Automated Planning for Adaptive Behavior (Extended Version)
- **Authors**: Daniel Ciolek, Nicolás D'Ippolito, Alberto Pozanco, Sebastian Sardina
- **Published**: 2020-02-27T21:16:01Z
- **Link**: [http://arxiv.org/abs/2002.12445v1](http://arxiv.org/abs/2002.12445v1)
- **Summary**: This paper proposes a multi-tier framework for automated planning that can handle different sets of assumptions and objectives. This could be relevant for making our `synthesis_planner` more robust and able to generate alternative plans based on different constraints or desired outcomes.

---

### Re-evaluating Retrosynthesis Algorithms with Syntheseus
- **Authors**: Krzysztof Maziarz, et al.
- **Published**: 2023-10-30T17:59:04Z
- **Link**: [http://arxiv.org/abs/2310.19796v3](http://arxiv.org/abs/2310.19796v3)
- **Summary**: The authors present "syntheseus," a library for benchmarking synthesis planning algorithms. They argue for more consistent and meaningful evaluation. While we are not developing a new algorithm from scratch, this paper provides valuable insights into how to evaluate the performance of our `synthesis_planner`.

---

### Automated Synthesis of Steady-State Continuous Processes using Reinforcement Learning
- **Authors**: Quirin Göttl, Dominik G. Grimm, Jakob Burger
- **Published**: 2021-01-12T11:49:34Z
- **Link**: [http://arxiv.org/abs/2101.04422v2](http://arxiv.org/abs/2101.04422v2)
- **Summary**: This work demonstrates how reinforcement learning can be used for automated flowsheet synthesis for continuous processes. The RL agent learns to build flowsheets in a simulator. This is a very promising direction for the `synthesis_planner`, suggesting a move towards more adaptive, learning-based approaches.

---

## 5. Quantum Error Correction

As the scale and complexity of simulations in the `quantum_lab` grow, understanding and eventually simulating quantum error correction (QEC) will be important for modeling realistic noisy quantum computers.

---

### Algebraic and information-theoretic conditions for operator quantum error-correction
- **Authors**: Michael A. Nielsen, David Poulin
- **Published**: 2005-06-09T00:45:04Z
- **Link**: [http://arxiv.org/abs/quant-ph/0506069v1](http://arxiv.org/abs/quant-ph/0506069v1)
- **Summary**: This paper provides a foundational framework for operator quantum error-correction, which unifies standard QEC with concepts like decoherence-free subspaces. The algebraic and information-theoretic conditions developed here could inform the theoretical underpinnings of any QEC simulation module in QuLab Infinite.

---

### Learning time-dependent noise to reduce logical errors: Real time error rate estimation in quantum error correction
- **Authors**: Ming-Xia Huo, Ying Li
- **Published**: 2017-10-10T14:56:23Z
- **Link**: [http://arxiv.org/abs/1710.03636v2](http://arxiv.org/abs/1710.03636v2)
- **Summary**: The authors propose a protocol for real-time monitoring of error rates in a QEC process without interrupting it. Using a Gaussian processes algorithm to estimate and predict error rates, they show a significant reduction in error correction failures. This suggests a dynamic, adaptive approach to simulating noise in the `quantum_lab`.

---

### An Introduction to Error-Correcting Codes: From Classical to Quantum
- **Authors**: Hsun-Hsien Chang
- **Published**: 2006-02-18T03:17:50Z
- **Link**: [http://arxiv.org/abs/quant-ph/0602157v1](http://arxiv.org/abs/quant-ph/0602157v1)
- **Summary**: This survey provides a good overview of the parallels between classical and quantum error-correcting codes. It serves as a good introductory resource for the fundamental concepts of QEC, which would be essential for anyone implementing QEC simulations.

---

### Demonstration of teleportation-based error correction in the IBM quantum computer
- **Authors**: K. M. Anandu, Muhammad Shaharukh, Bikash K. Behera, Prasanta K. Panigrahi
- **Published**: 2019-02-02T08:03:10Z
- **Link**: [http://arxiv.org/abs/1902.01692v1](http://arxiv.org/abs/1902.01692v1)
- **Summary**: This paper demonstrates a teleportation-based error correction (TEC) protocol on an IBM quantum computer. While QuLab Infinite is a simulator, modeling specific, hardware-realized QEC protocols like this one could be a valuable validation and benchmarking tool for the `quantum_validation` module.

---

### A note on threshold theorem of fault-tolerant quantum computation
- **Authors**: Min Liang, Li Yang
- **Published**: 2010-06-25T09:00:34Z
- **Link**: [http://arxiv.org/abs/1006.4941v1](http://arxiv.org/abs/1006.4941v1)
- **Summary**: The threshold theorem is a cornerstone of fault-tolerant quantum computation. This note discusses the optimal period for applying error correction, showing it depends on the level of concatenation in the code. This is a subtle but important detail that could influence the design of long-running, fault-tolerant simulations.

---

## 6. Tensor Network Quantum Simulation

Tensor networks provide a way to simulate quantum systems with more than 30-50 qubits, which is a stated goal for QuLab Infinite's `quantum_lab`. These methods are particularly effective for systems with limited entanglement.

---

### Simplification of tensor updates toward performance-complexity balanced quantum computer simulation
- **Authors**: Koichi Yanagisawa, Aruto Hosaka, Tsuyoshi Yoshida
- **Published**: 2024-06-05T07:18:28Z
- **Link**: [http://arxiv.org/abs/2406.03010v1](http://arxiv.org/abs/2406.03010v1)
- **Summary**: This paper studies the simplification of tensor updates in tensor network simulations. It highlights the "simple update" method as providing a good balance between fidelity and computational complexity. This could be a practical technique to implement for our tensor network simulator to improve performance.

---

### Tensor Networks for Simulating Quantum Circuits on FPGAs
- **Authors**: Maksim Levental
- **Published**: 2021-08-15T22:43:38Z
- **Link**: [http://arxiv.org/abs/2108.06831v1](http://arxiv.org/abs/2108.06831v1)
- **Summary**: While focused on FPGAs, this work explores how tensor networks can reduce the memory footprint of quantum circuit simulations. The general principles of identifying economical tensor contractions are directly applicable to a software-based simulator in QuLab Infinite.

---

### Local tensor network for strongly correlated projective states
- **Authors**: B. Béri, N. R. Cooper
- **Published**: 2011-01-28T19:15:07Z
- **Link**: [http://arxiv.org/abs/1101.5610v1](http://arxiv.org/abs/1101.5610v1)
- **Summary**: The authors show how to encode calculations for strongly correlated projective states (like fractional quantum Hall states) in a local Grassmann tensor network. This is a specialized but powerful technique that could extend the range of quantum materials simulatable in `quantum_materials.py`.

---

### Tensor Network States with Low-Rank Tensors
- **Authors**: Hao Chen, Thomas Barthel
- **Published**: 2022-05-30T17:58:16Z
- **Link**: [http://arxiv.org/abs/2205.15296v1](http://arxiv.org/abs/2205.15296v1)
- **Summary**: This paper introduces the idea of imposing low-rank constraints on the tensors within a tensor network. This can substantially reduce the time and space complexity of simulations. This is a promising optimization for the tensor network backend of our `quantum_lab`.

---

### Harnessing CUDA-Q's MPS for Tensor Network Simulations of Large-Scale Quantum Circuits
- **Authors**: Gabin Schieffer, Stefano Markidis, Ivy Peng
- **Published**: 2025-01-27T10:36:05Z
- **Link**: [http://arxiv.org/abs/2501.15939v1](http://arxiv.org/abs/2501.15939v1)
- **Summary**: This work evaluates the use of Matrix Product State (MPS) tensor networks for large-scale quantum circuit simulation. It highlights the potential of tensor networks to simulate circuits with large numbers of qubits, which is a key goal for QuLab Infinite.

---

## 7. AI for Drug Discovery

The `chemistry_lab` has a strong focus on molecular dynamics and synthesis. AI-driven drug discovery is a natural extension of these capabilities and aligns with the project's goals.

---

### Accelerating drug discovery with Artificial: a whole-lab orchestration and scheduling system for self-driving labs
- **Authors**: Yao Fehlis, Paul Mandel, Charles Crain, Betty Liu, David Fuller
- **Published**: 2025-04-01T17:22:50Z
- **Link**: [http://arxiv.org/abs/2504.00986v1](http://arxiv.org/abs/2504.00986v1)
- **Summary**: This paper describes a lab orchestration system for AI-guided experimentation in drug discovery. This aligns well with the `hive_mind` concept and provides a model for how to integrate AI/ML models (like those for molecular interaction prediction) into an automated experimental workflow.

---

### Artificial Intelligence for Drug Discovery: Are We There Yet?
- **Authors**: Catrin Hasselgren, Tudor I. Oprea
- **Published**: 2023-07-13T01:51:26Z
- **Link**: [http://arxiv.org/abs/2307.06521v1](http://arxiv.org/abs/2307.06521v1)
- **Summary**: This review discusses the use of AI in diseases, targets, and therapeutics. It covers generative chemistry, machine learning, and multi-property optimization. This is a good high-level overview of the field and can help guide the development of new features in the `chemistry_lab`.

---

### CardiGraphormer: Unveiling the Power of Self-Supervised Learning in Revolutionizing Drug Discovery
- **Authors**: Abhijit Gupta
- **Published**: 2023-07-03T08:58:32Z
- **Link**: [http://arxiv.org/abs/2307.00859v4](http://arxiv.org/abs/2307.00859v4)
- **Summary**: The author introduces CardiGraphormer, which combines self-supervised learning, GNNs, and a special attention mechanism to learn molecular representations. This could be a powerful tool for developing new property predictors in `ml_property_predictor.py` or for guiding synthesis in `synthesis_planner.py`.

---

### Artificial Intelligence Approaches for Anti-Addiction Drug Discovery
- **Authors**: Dong Chen, Jian Jiang, Zhe Su, Guo-Wei Wei
- **Published**: 2025-02-05T20:49:02Z
- **Link**: [http://arxiv.org/abs/2502.03606v2](http://arxiv.org/abs/2502.03606v2)
- **Summary**: This review explores the role of AI in the anti-addiction drug discovery pipeline, from data collection to compound optimization. While the application is specific, the AI techniques described are broadly applicable to drug discovery in general.

---

### ChatGPT in Drug Discovery: A Case Study on Anti-Cocaine Addiction Drug Development with Chatbots
- **Authors**: Rui Wang, Hongsong Feng, Guo-Wei Wei
- **Published**: 2023-08-14T03:43:57Z
- **Link**: [http://arxiv.org/abs/2308.06920v2](http://arxiv.org/abs/2308.06920v2)
- **Summary**: This paper showcases using GPT-4 as a virtual guide for drug discovery. The chatbot offers strategic and methodological insights. This is a fascinating example of how LLMs could be integrated into the `hive_mind` to assist with experiment design and planning.

---

## 8. Machine Learning for Computational Fluid Dynamics (CFD)

The `physics_engine` includes a fluid dynamics component. Machine learning can be used to accelerate CFD simulations, create reduced-order models, and improve turbulence modeling.

---

### Enhancing Computational Fluid Dynamics with Machine Learning
- **Authors**: Ricardo Vinuesa, Steven L. Brunton
- **Published**: 2021-10-05T14:34:16Z
- **Link**: [http://arxiv.org/abs/2110.02085v2](http://arxiv.org/abs/2110.02085v2)
- **Summary**: This perspective article highlights key areas where ML can impact CFD, including accelerating simulations, improving turbulence closure modeling, and developing reduced-order models. This provides a roadmap for potential ML integration into `physics_engine/fluid_dynamics.py`.

---

### Machine learning in fluid dynamics: A critical assessment
- **Authors**: Kunihiko Taira, Georgios Rigas, Kai Fukami
- **Published**: 2025-08-19T01:06:05Z
- **Link**: [http://arxiv.org/abs/2508.13430v2](http://arxiv.org/abs/2508.13430v2)
- **Summary**: This article offers a critical assessment of the challenges and opportunities for ML in fluid dynamics. It emphasizes the need for community-maintained datasets and open-source code, which is a good reminder for the QuLab Infinite project as a whole.

---

### Learning Incompressible Fluid Dynamics from Scratch -- Towards Fast, Differentiable Fluid Models that Generalize
- **Authors**: Nils Wandel, Michael Weinmann, Reinhard Klein
- **Published**: 2020-06-15T20:59:28Z
- **Link**: [http://arxiv.org/abs/2006.08762v3](http://arxiv.org/abs/2006.08762v3)
- **Summary**: The authors propose a physics-constrained training approach for CNNs to simulate fluids. The models generalize to new domains and don't require simulation data for training. This could be a very powerful way to create fast, surrogate models for the fluid dynamics simulations in QuLab Infinite.

---

### Inpainting Computational Fluid Dynamics with Deep Learning
- **Authors**: Dule Shu, Wilson Zhen, Zijie Li, Amir Barati Farimani
- **Published**: 2024-02-27T03:44:55Z
- **Link**: [http://arxiv.org/abs/2402.17185v1](http://arxiv.org/abs/2402.17185v1)
- **Summary**: This paper tackles the problem of completing fluid data (inpainting) using deep learning. This could be used to reconstruct fluid flow fields from sparse sensor data or to fill in missing data from coarse simulations, enhancing the capabilities of the `environmental_sim`.

---

### Parameter-Conditioned Sequential Generative Modeling of Fluid Flows
- **Authors**: Jeremy Morton, Freddie D. Witherden, Mykel J. Kochenderfer
- **Published**: 2019-12-14T00:16:53Z
- **Link**: [http://arxiv.org/abs/1912.06752v1](http://arxiv.org/abs/1912.06752v1)
- **Summary**: This paper introduces a method for learning neural network models that can perform parameterized simulations of fluid flows. The models can generate simulations for a range of flow conditions orders of magnitude faster than traditional CFD. This is another excellent candidate for creating surrogate models in the `physics_engine`.

# EdgeAIBus: AI-driven Joint Container Management and Model Selection Framework for Heterogeneous Edge Computing

Containerized Edge computing offers lightweight, reliable, and quick solutions to latency-critical Machine Learning (ML) and Deep Learning (DL) applications. Existing solutions considering multiple Quality of Service (QoS) parameters either overlook the intricate relation of QoS parameters or pose significant scheduling overheads. Furthermore, reactive decision-making can damage Edge servers at peak load, incurring escalated costs and wasted computations. Resource provisioning, scheduling, and ML model selection substantially influence energy consumption, user-perceived accuracy, and delay-oriented Service Level Agreement (SLA) violations. Addressing contrasting objectives and QoS simultaneously while avoiding server faults is highly challenging in the exposed heterogeneous and resource-constrained Edge continuum. In this work, we propose the EdgeAIBus framework that offers a novel joint container management and ML model selection algorithm based on Importance Weighted Actor-Learner Architecture to optimize energy, accuracy, SLA violations, and avoid server faults. Firstly, Patch Time Series Transformer (PatchTST) is utilized for CPU usage predictions of Edge servers for its 8.51% Root Mean Squared Error and 5.62% Mean Absolute Error. Leveraging pipelined predictions, EdgeAIBus conducts consolidation, resource oversubscription, and ML/DL model switching with possible migrations to conserve energy, maximize utilization and user-perceived accuracy, and reduce SLA violations. Simulation results show EdgeAIBus oversubscribed 110% cluster-wide CPU with real usage up to 70%, conserved 14 CPU cores, incurred less than 1% SLA violations with 2.54% drop in inference accuracy against industry-led Model Switching Balanced load and Google Kubernetes Optimized schedulers. Google Kubernetes Engine experiments demonstrate 80% oversubscription, 14 CPU cores conservation, 1% SLA violations, and 3.81% accuracy loss against the counterparts. Finally, constrained setting experiment analysis shows that PatchTST and EdgeAIBus can produce decisions within 100ms in a 1-core and 1 GB memory device.


<p align="center">
  <img src="https://github.com/user-attachments/assets/4b9af656-5176-4f1a-83a4-b8f5ccbd45ca" alt="EdgeAIBus">
</p>

# Cite this work
```bibtex
@ARTICLE{ali2025edgeaibus,
  author={Ali, Babar and Golec, Muhammed and Gill, Sukhpal Singh and Cuadrado, Felix and Uhlig, Steve},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={EdgeAIBus: AI-driven Joint Container Management and Model Selection Framework for Heterogeneous Edge Computing}, 
  year={2025},
  pages={1-12},
  doi={10.1109/TPDS.2025.3602521}}
```

# License
BSD-3-Clause. Copyright (c) 2025, Babar Ali. All rights reserved.
See the [License](https://github.com/BabarAli93/EdgeAIBus/blob/main/LICENSE) file for more details.

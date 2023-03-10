
# Upcycling Models under Domain and Category Shift [CVPR-2023]

## Introduction
Deep neural networks (DNNs) often perform poorly in the presence of domain shift and category shift. To address this, in this paper, we explore the Source-free Universal Domain Adaptation (SF-UniDA). SF-UniDA is appealing in view that universal model adaptation can be resolved only on the basis of a standard pre-trained closed-set model, i.e., without source raw data and dedicated model architecture. To achieve this, we develop a generic global and local clustering technique (GLC). GLC equips with an inovative one-vs-all global pseudo-labeling strategy to realize "known" and "unknown" data samples separation under various category-shift. Remarkably, in the most challenging open-partial-set DA scenario, GLC outperforms UMAD by 14.8% on the VisDA benchmark.

<img src="figures/SFUNIDA.png" width="500"/>

## Framework
<img src="figures/GLC_framework.png" width="1000"/>

## Prerequisites
- python3, pytorch, numpy, PIL, scipy, sklearn, tqdm, etc.
- We have presented the our conda environment file in `./environment.yml`.

## Dataset
We have conducted extensive expeirments on four datasets with three category shift scenario, i.e., Partial-set DA (PDA), Open-set DA (OSDA), and Open-partial DA (OPDA). The following is the details of class split for each scenario. Here, $\mathcal{Y}$, $\mathcal{\bar{Y}_s}$, and $\mathcal{\bar{Y}_t}$ denotes the source-target-shared class, the source-private class, and the target-private class, respectively. 

| Datasets    | Class Split| $\mathcal{Y}/\mathcal{\bar{Y}_s}/\mathcal{\bar{Y}_t}$| |
| ----------- | --------   | -------- | -------- |
|     | OPDA       | OSDA     | PDA      |
| Office-31   | 10/10/11   | 10/0/11  | 10/21/0  |
| Office-Home | 10/5/50    | 25/0/40  | 25/40/0  |
| VisDA-C     | 6/3/3      | 6/0/6    | 6/6/0    |
| DomainNet   | 150/50/145 |          |          |

Please manually download these datasets from the official websites, and unzip them to the `./data` folder. To ease your implementation, we have provide the `image_unida_list.txt` for each dataset subdomains. 

```
./data
├── Office
│   ├── Amazon
|       ├── ...
│       ├── image_unida_list.txt
│   ├── Dslr
|       ├── ...
│       ├── image_unida_list.txt
│   ├── Webcam
|       ├── ...
│       ├── image_unida_list.txt
├── OfficeHome
│   ├── ...
├── VisDA
│   ├── ...
```

## Training
1. Open-partial Domain Adaptation (OPDA) on Office, OfficeHome, and VisDA
```
cd ./scripts
# Source Model Preparing
bash ./scripts/train_source_OPDA.sh
# Target Model Adaptation
bash ./scripts/train_target_OPDA.sh
```
2. Open-set Domain Adaptation (OSDA) on Office, OfficeHome, and VisDA
```
cd ./scripts
# Source Model Preparing
bash ./scripts/train_source_OSDA.sh
# Target Model Adaptation
bash ./scripts/train_target_OSDA.sh
```
3. Partial-set Domain Adaptation (PDA) on Office, OfficeHome, and VisDA
```
cd ./scripts
# Source Model Preparing
bash ./scripts/train_source_PDA.sh
# Target Model Adaptation
bash ./scripts/train_target_PDA.sh
```

## Citation
If you find our codebase helpful, please star our project and cite our paper:
```
@inproceedings{sanqing2023GLC,
  title={Upcycling Models under Domain and Category Shift},
  author={Qu, Sanqing and Zou, Tianpei and Röhrbein, Florian and Lu, Cewu and Chen, Guang and Tao, Dacheng and Jiang, Changjun},
  booktitle={CVPR},
  year={2023},
}

@inproceedings{sanqing2022BMD,
  title={BMD: A general class-balanced multicentric dynamic prototype strategy for source-free domain adaptation},
  author={Qu, Sanqing and Chen, Guang and Zhang, Jing and Li, Zhijun and He, Wei and Tao, Dacheng},
  booktitle={ECCV},
  year={2022}
}
```

## Contact
- sanqingqu@gmail.com or 2011444@tongji.edu.cn

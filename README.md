# AI-Human Collabration Bronchoscope Robot for Foreign Body Aspiration Surgery 

The official PyTorch implementation of our paper "AI Search, Physician Removal: A Low-cost 5G Bronchoscopy Robot Enables Collaboration in Foreign Body Aspiration"

Video:
[![AI Search, Physician Removal: A Low-cost 5G Bronchoscopy Robot Enables Collaboration in FBA](https://img.youtube.com/vi/cimMRYJC7xk/maxresdefault.jpg)](https://youtu.be/cimMRYJC7xk?si=RBZLCcQCRZAKQPJ4 "AI Search, Physician Removal: A Low-cost 5G Bronchoscopy Robot Enables Collaboration in FBA")

## Introduction

Bronchial foreign body aspiration is a life-threatening condition with a high incidence across diverse populations, requiring urgent diagnosis and treatment. However, the limited availability of skilled practitioners and advanced medical equipment in community clinics and underdeveloped regions underscores the broader challenges in emergency care. Here, we present a cost-effective robotic bronchoscope capable of CT-free, artificial intelligence (AI)-driven foreign body search and doctor-collaborated removal over long distances via the 5th-generation (5G) communication. The system is built around a low-cost (< 5000 USD), portable (< 2 kg) bronchoscope robotic platform equipped with a 3.3 mm diameter catheter and 1 mm biopsy forceps, designed for safe pulmonary search and foreign body removal. Our AI algorithm, which integrates classical data structures with modern machine learning techniques, enables thorough CT-free lung coverage. The tree structure is leveraged to memorize a compact exploration process, and guide the decision-making. Both virtual and physical simulations demonstrate the system’s superior autonomous foreign body search, minimizing bronchial wall contact to reduce patient discomfort. In a remote procedure, a physician in Hangzhou successfully retrieved a foreign body from a live pig located 1500 km away in Chengdu using 5G communication, highlighting effective collaboration between AI, robotics, and human experts. We anticipate that this 5G-enabled, low-cost, AI-expert-collaborated robotic platform has significant potential to reduce medical disparities, enhance emergency care, improve patient outcomes, decrease physician workload, and streamline medical procedures through the automation of routine tasks.

<img src="figs/teaser.png#pic_left" alt="avatar" style="zoom:40%;" />

## Usage

### Prerequisites
* Python 3.7.12
* PyTorch 1.13.1 and torchvison (https://pytorch.org/)
* torchvision 0.14.1
* VTK 8.1.2
* Pyrender 0.1.45
* PyBullet 3.2.6
* CUDA 12.3


### Installation

Necessary Python packages can be installed by

```bash
pip install -r requirements.txt
```

### Train
```
> python train.py  --dataset-dir YOUR_AIRWAY_DIR
```
The training dataset can be got by emailing us with reasonable requrest.

### Test
The testing environment of Patient 3 and network model trained on Patient 1 and 2 can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1g0YX9uB9_yNnwHU1synubgy7KFsg4syx?usp=drive_link). Save /Airways, /checkpoints_policy and /checkpoints_LD folders in root direction of the repo, then type the following code for evaulating the Sampled method:

```
> python test.py
```
The testing will start after close the bronchial tree figure shown by NetworkX and Matplotlib.

<img src="figs/comb.png#pic_left" alt="avatar" style="zoom:40%;" />

## Results

### Simulation

<img src="figs/simulation.png#pic_left" alt="avatar" style="zoom:100%;" />

### In-vitro
<img src="figs/invitro.png#pic_left" alt="avatar" style="zoom:100%;" />

### In-vivo
<img src="figs/invivo.png#pic_left" alt="avatar" style="zoom:100%;" />

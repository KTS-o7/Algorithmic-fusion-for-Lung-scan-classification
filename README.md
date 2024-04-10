# Algorithmic Fusion for Lung Scan Classification

## Overview

![GitHub Languages](https://img.shields.io/github/languages/count/KTS-o7/Algorithmic-fusion-for-Lung-scan-classification)
![GitHub Last Commit](https://img.shields.io/github/last-commit/KTS-o7/Algorithmic-fusion-for-Lung-scan-classification)
![GitHub Contributors](https://img.shields.io/github/contributors/KTS-o7/Algorithmic-fusion-for-Lung-scan-classification)
![GitHub top language](https://img.shields.io/github/languages/top/KTS-o7/Algorithmic-fusion-for-Lung-scan-classification)
![GitHub](https://img.shields.io/github/license/KTS-o7/Algorithmic-fusion-for-Lung-scan-classification)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FKTS-o7%2FAlgorithmic-fusion-for-Lung-scan-classification&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false)](https://hits.seeyoufarm.com)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Author](#author)
- [Contact](#contact)

## Introduction

Algorithmic Fusion for Lung Scan Classification is a research project focused on developing and evaluating fusion techniques for improving the accuracy of lung disease classification from medical images. By combining the strengths of multiple deep learning models, the project aims to enhance the performance of traditional classification approaches.

### Models Used

The project incorporates several powerful deep learning models for lung disease classification, including:

- GoogleNet
- DenseNet
- VGG19
- MobileNet
- ResNet50

## Objectives

The primary objectives of the Algorithmic Fusion for Lung Scan Classification project include:

1. Investigate the effectiveness of algorithmic fusion techniques in combining features extracted from diverse deep learning models.
2. Develop a comprehensive classification system capable of accurately distinguishing between different lung diseases, including COVID-19, pneumonia, and normal cases.
3. Evaluate the performance of individual deep learning models (such as ResNet, VGG, Googlenet, MobileNet, and DenseNet) and their fused combinations in terms of accuracy, precision, recall, and F1-score.
4. Determine scenarios where algorithmic fusion yields superior results compared to using single deep learning models alone.
5. Provide insights into the trade-offs and benefits of employing fusion techniques for lung scan classification tasks.

## Installation

To use the Algorithmic Fusion for Lung Scan Classification project, follow these steps:

1. **Clone the Repository**: Clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/KTS-o7/Algorithmic-fusion-for-Lung-scan-classification.git
   ```

2. **Install Dependencies**: Navigate to the project directory and install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the Code**: Dive into the project codebase to understand the implementation of fusion techniques and classification algorithms.

## Usage

**Important Note**: This project involves computationally expensive tasks, and the use of a GPU is highly recommended for efficient execution.

- Obtain the dataset for the model from [Dataset Link](https://www.kaggle.com/datasets/anaselmasry/covid19normalpneumonia-ct-images?select=COVID2_CT).
- The [`training.ipynb`](./SingularTraining.ipynb) file contains code to train individual models used in the project. Uncomment the desired model and comment out others in the model selection cell. Repeat the same changes in the training function call cell.
- The [`multimodel.ipynb`](./Multimodel.ipynb) file is the core of the project, where model fusion is performed. Substitute the desired model names and make necessary code adjustments for fusion.
- Use the [`evaluate.ipynb`](./Evalutaion.ipynb) file to validate and evaluate the trained models.

Alternatively, you can run the project on Google Colab.
To run the project on Google Colab, follow these steps:

1. Go to Google Colab: [Google Colab](https://colab.research.google.com/).
2. Download the project files from the repository.
3. Upload the files to Google Colab one by one.
4. Change the runtime environment to GPU for better performance.
5. Run the cells in the uploaded notebooks to execute the project.

The project includes several Jupyter Notebook files for different aspects of the research, including model training, fusion techniques, and evaluation. Here's a brief overview of each file:

## Author

[KTS-o7](https://github.com/KTS-o7)

## Contact

For any inquiries or support, please contact [Krishnatejaswi S](mailto:shentharkrishnatejaswi@gmail.com).

## Citation:

Please cite this project in your publications or presentations as follows:

### APA Style

```
Shenthar, K. (2024). Algorithmic-fusion-for-Lung-scan-classification. GitHub. https://github.com/KTS-o7/Algorithmic-fusion-for-Lung-scan-classification.git
```

### MLA Style

```
Shenthar, Krishnatejaswi. "Algorithmic-fusion-for-Lung-scan-classification." GitHub, 2024, https://github.com/KTS-o7/Algorithmic-fusion-for-Lung-scan-classification.git.
```

### Chicago Style

```
Shenthar, Krishnatejaswi. "Algorithmic-fusion-for-Lung-scan-classification." GitHub, 2024.
```

### JSON Style

```json
{
  "title": "Algorithmic Fusion for Lung Scan Classification",
  "author": "Krishnatejaswi Shenthar",
  "year": 2024,
  "repository": "Algorithmic-fusion-for-Lung-scan-classification",
  "url": "https://github.com/KTS-o7/Algorithmic-fusion-for-Lung-scan-classification.git",
  "license": "MIT",
  "description": "A research project focused on developing and evaluating fusion techniques for improving the accuracy of lung disease classification from medical images."
}
```

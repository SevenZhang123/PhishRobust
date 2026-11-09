# PhishRobust :fishing_pole_and_fish:
_This repository contains the code, data, models, and reproduction methods for **PhishRobust**!_

> [!IMPORTANT]
> You can choose two methods to reproduce the content:
> 
> **1. One-click Run Version:** Based on our pre-trained models, run PhishRobust with a single click for convenient and quick evaluation and testing.
> 
> **2. Self-built Version:** Based on the code of each module, train the corresponding model according to your own needs, and then run PhishRobust for evaluation and testing.

## Environment Preparation
Please install the following Python packages in advance:
```
 1. torch --- 2.8.0 or above
 2. censys --- 2.2.16 or above
```

## One-click Run Version
(1). Based on the model addresses in `model_path`, download those models to the `models` folder.

(2). Run `Sample_label_construction.py` using the following command to build the sample-label mapping file (`sample_label.txt`):
```
 cd code
 python Sample_label_construction.py
```
* `Sample_label_construction.py`: Construct the sample-label mapping file (`sample_label.txt`).

(3). Run `PhishRobust.py` using the following command:
```
 cd code
 python PhishRobust.py
```
* `PhishRobust.py`: PhishRobust model full-process framework.

## Self-built Version

_In the self-built Version, we need to complete the overall reproduction through the following five steps._

**Step-1: Logo Localization and Extraction** 

(1). Run `convert_coco_to_yolo.py` using the following command to convert the raw data into a YOLO-compatible format:
```
 cd code/Logo_Localizer
 python convert_coco_to_yolo.py
```
* `convert_coco_to_yolo.py`: Convert COCO format datasets to YOLO format, compatible with YOLO11_Detect training and testing.

(2). Run `train_yolo.py` using the following command to train the YOLO11_Detect model based on the newly transformed data. The trained model (`logo_location_and_select.pt`) is located in `runs/train/yolo_logo_exp1/weights`. Please move it to the `models` directory:
```
 cd code/Logo_Localizer
 python train_yolo.py
```
* `train_yolo.py`: Training the YOLO11_Detect model on a newly transformed dataset.

**Step-2: Robust Encoder Construction**

(1). Download `Logo-2K+` and `target_list_134` datasets, and place them in the `data` directory.

(2). Run `Classifier_pre-training.py` using the following commands. We use the `Logo-2K+` dataset to initially train the classifier. The trained initial classifier (`initial_classifier.pth`) is located in the `models` folder:
```
 cd code/Robust_encoder_construction
 python Classifier_pre-training.py
```
* `Classifier_pre-training.py`: Perform initial training to obtain an initial classifier.

(3). Run `FD-PGD_Adversarial_Training_first_phase.py` using the following commands. We use the `Logo-2K+` dataset to perform the first-stage adversarial training on the initial classifier. The trained first-stage robust classifier (`robust_classifier_first_phase.pth`) is located in the `models` folder:
```
 cd code/Robust_encoder_construction
 python FD-PGD_Adversarial_Training_first_phase.py
```
* `FD-PGD_Adversarial_Training_first_phase.py`: Perform the first stage of adversarial training on the initial classifier.

(4). Run `FD-PGD_Adversarial_Training_second_phase.py` using the following commands. We use the `target_list_134` dataset to perform second-stage adversarial training on the classifier. The trained second-stage robust classifier (`robust_classifier_second_phase.pth`) is located in the `models` folder:
```
 cd code/Robust_encoder_construction
 python FD-PGD_Adversarial_Training_second_phase.py
```
* `FD-PGD_Adversarial_Training_second_phase.py`: Perform the second stage of adversarial training on the robust classifier obtained from the first stage of adversarial training.

(5). Run `Robust_logo_encoder.py` using the following command. It uses transfer learning to extract the classification head of the robust classifier and build a robust logo encoder (`robust_logo_encoder.pth`):
```
 cd code/Robust_encoder_construction
 python Robust_logo_encoder.py
```
* `Robust_logo_encoder.py`: Build the robust logo encoder based on transfer learning.

**Step-3: Brand Logo Feature Distribution Construction**

(1). Run `logo_sample_expand.py` using the following command to expand the number of official logos for the brand, preparing for the next step of building the feature distribution:
```
 cd code/Feature_distribution_construction
 python logo_sample_expand.py
```
* `logo_sample_expand.py`: Expanding the brand's official logo based on traditional computer vision methods.

(2). Run the command `logo_feature_distribution_build.py` to build the logo feature distribution for each brand. The built feature distribution library (`brand_logo_feature_distribution.pkl`) is located in the `models` folder:
```
 cd code/Feature_distribution_construction
 python logo_feature_distribution_build.py
```
* `logo_feature_distribution_build.py`: Construct the logo feature distribution for each brand.

**Step-4: Domain List Construction based on Active Probing**

(1). Run `Active_probing.py` using the following commands to probe the official domain of each brand through Censys' API and build a domain list. The built domain list for each brand are located in the `data/brand_domain` folder:
```
 cd code/Domain_list_construction
 python Active_probing.py
```
* `Active_probing.py`: Domain list construction based on active probing.

**Step-5: PhishRobust Process**

(1). Run `Sample_label_construction.py` using the following command to build the sample-label mapping file (`sample_label.txt`):
```
 cd code
 python Sample_label_construction.py
```
* `Sample_label_construction.py`: Construct the sample-label mapping file (`sample_label.txt`).

(2). Run `PhishRobust.py` using the following command:
```
 cd code
 python PhishRobust.py
```
* `PhishRobust.py`: PhishRobust model full-process framework. 

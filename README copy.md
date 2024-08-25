# COVID-19 Segmentation Using Deep Learning

## Introduction

In the fight against COVID-19, precise and prompt patient diagnosis is essential for managing the disease and preventing its spread. A recent study investigated the use of transfer learning with various architectures like Densenet, Gernet, and SeNet, alongside decoder models such as Unet++, Deeplabv3, and Deeplabv3+, to identify pulmonary and COVID-19 infection regions. Using a dataset from the Italian Society of Medical and Interventional Radiology (SIRM), which included both positive and negative samples, Unet++ with Densenet161 achieved the best results in specificity, sensitivity, Dice coefficient, and IoU, with scores of 87.6%, 91.7%, 89.6%, and 81.1%, respectively. These advancements are expected to enhance diagnostic accuracy and efficiency, helping doctors to save time, reduce costs, and better manage the pandemic.

## Dataset

The dataset utilized in this study consists of annotated lung CT scans from COVID-19 patients. The scans are accompanied by ground truth masks indicating the infected regions. The dataset is publicly available and can be accessed from [zenodo dataset](https://zenodo.org/records/3757476).

## Methodology

### Model Architecture

We employ a modified U-Net++ architecture for segmentation. The U-Net++ model consists of an encoder-decoder structure with skip connections, allowing for efficient localization and precise segmentation. The encoder captures context through successive downsampling, while the decoder reconstructs the segmentation map through upsampling.


### Training

The model is trained using a combination of binary cross-entropy and Dice loss, which helps in handling class imbalance and improving segmentation accuracy. The training process is carried out using the Adam optimizer with a learning rate of 1e-4. The model is trained for 100 epochs with a batch size of 2.

```sh
## Implementation

### training

python trainval.py --encoder densenet201 --base unetplus --datadir /dataset --savedir_base ./figures_no_augment 

### Testing

python test.py --datadir /dataset --savedir_base /figures_no_augment -ei unetplus_densenet201


```sh
pip install -r requirements.txt

# KnowSAM
Official code for "[Learnable Prompting SAM-induced Knowledge Distillation for Semi-supervised Medical Image Segmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10843257)"

## Installation

To set up the environment and install dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset

We provide Google Drive access links to the datasets employed in this study, among which are two open-source datasets cited in the corresponding paper:

[Endoscope Dataset](https://drive.google.com/drive/folders/1uxPGLAon7fTH2qbnohRYOZomApKs9wjS?hl=zh)

[BCSS Dataset](https://drive.google.com/file/d/1Xig3I0rBG9Te3wh8ZaFeQKav-bUxf5YL/view?usp=drive_link)

## Extract Sample Data

We provide a reference sample dataset (SampleData.rar) that allows users to quickly test and run the model. Extract the dataset using the following command:
```bash
unrar x SampleData.rar
```
For processed ACDC dataset, you can download it from the [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC), and place it directly in the `SampleData` folder.


## Training
To train the model on a dataset, execute:
```bash
python train_semi_SAM.py
```

For ACDC dataset training:
```bash
python train_semi_SAM_ACDC.py
```

## Prediction
After training, you can make predictions using:
```bash
python prediction.py
```

For ACDC dataset inference:
```bash
python prediction_ACDC.py
```

## Acknowledgements
Our code is based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).

## Questions
If you have any questions, welcome contact me at 'taozhou.dreams@gmail.com'

# NIRS Brain Data Prediction
The competition hosted by DACON(https://dacon.io/competitions/official/235608/overview/) was open from 15.05.2020 to 26.06.2020 and attracted about 690 teams. The aim of the data analysis is to predict multiple targets, the concentrations of Hhb, HbO2, Ca and Na, by near-infrared spectroscopy(NIRS) data. The evaluation matric is MAE.

This code consists of three parts; feature engineering(dataset.py), LightGBM models(model.py) and implementation(main.py). 

## What is NIRS?(https://en.wikipedia.org/wiki/Near-infrared_spectroscopy)
Near-infrared spectroscopy (NIRS) is a spectroscopic method that uses the near-infrared region of the electromagnetic spectrum. Typical applications include medical and physiological diagnostics and research including blood sugar, pulse oximetry, functional neuroimaging, sports medicine, elite sports training, ergonomics, rehabilitation, neonatal research, brain computer interface, urology (bladder contraction), and neurology (neurovascular coupling). There are also applications in other areas as well such as pharmaceutical, food and agrochemical quality control, atmospheric chemistry, combustion research and astronomy.

(사진)

## Dataset(https://dacon.io/competitions/official/235608/data/) 
The dataset has 10,000 instances in the train and test datasets respectively. The features are the distance of measurement, initial signals and measurement signals(650 to 990nm wavelength). The target values are the concentrations of Hhb, HbO2, Ca and Na. 

## Feature Engineering
The data includes a lot of NaN values so that an interpolation method is necessary. Pandas library provides several interpolations such as linear, quadratic, and cubic. This code uses cubic interpolations by empirical experiments. In order to improve model performance, outliers are removed and then a customized scaling formula is applied to reduce the amplitude gaps caused by measurement distances. Next, Fast Fourier Transform(https://en.wikipedia.org/wiki/Fast_Fourier_transform) is used to compute isotopic distributions, and other new features are generated or some features are eliminated for each target value. 

## LightGBM model(https://lightgbm.readthedocs.io/en/latest/Parameters.html)
The model is an ensemble model combined with different three different LightGBM models. The hyper-parameters are fixed through a grid search. The reason why a separated model each target is built although MultiOutputRegressor can be used is that the different feature engineering methods are applied for each target value.

## Conclusion

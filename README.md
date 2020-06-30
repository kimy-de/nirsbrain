# NIRS Brain Data Prediction
The competition hosted by DACON(https://dacon.io/competitions/official/235608/overview/) was open from 15.05.2020 to 26.06.2020 and attracted 316 teams. The aim of the data analysis is to predict multiple targets, the concentrations of Hhb, HbO2, Ca and Na, by near-infrared spectroscopy(NIRS) data. The evaluation matric is MAE.

This code consists of three parts; feature engineering(dataset.py), LightGBM models(model.py) and implementation(main.py). 

## What is NIRS?(https://en.wikipedia.org/wiki/Near-infrared_spectroscopy)
Near-infrared spectroscopy (NIRS) is a spectroscopic method that uses the near-infrared region of the electromagnetic spectrum. Typical applications include medical and physiological diagnostics and research including blood sugar, pulse oximetry, functional neuroimaging, sports medicine, elite sports training, ergonomics, rehabilitation, neonatal research, brain computer interface, urology (bladder contraction), and neurology (neurovascular coupling). There are also applications in other areas as well such as pharmaceutical, food and agrochemical quality control, atmospheric chemistry, combustion research and astronomy.

## Dataset(https://dacon.io/competitions/official/235608/data/) 
The dataset has 10,000 instances in the train and test datasets respectively. The features are the distance of measurement, initial signals and measurement signals(650 to 990nm wavelength). The target values are the concentrations of Hhb, HbO2, Ca and Na. 
![dist_targets](https://user-images.githubusercontent.com/52735725/86007620-f4112580-ba17-11ea-97c6-3401c79c4d13.png)
## Feature Engineering
The data includes a lot of NaN values so that an interpolation method is necessary. Pandas library provides several interpolations such as linear, quadratic, and cubic. This code uses cubic interpolations. In order to improve model performance, outliers are removed and then a customized scaling formula is applied to reduce the amplitude gaps caused by measurement distances. Next, Fast Fourier Transform(https://en.wikipedia.org/wiki/Fast_Fourier_transform) is used to compute isotopic distributions, and other new features are generated or some features are eliminated for each target value. 
![nan_values](https://user-images.githubusercontent.com/52735725/86008311-d98b7c00-ba18-11ea-9682-7385d4a57a6b.png)

## LightGBM model(https://lightgbm.readthedocs.io/en/latest/Parameters.html)
The model is an ensemble model combined with different three different LightGBM models. The hyper-parameters are fixed through a grid search. The reason why a separated model each target is built although MultiOutputRegressor can be used is that the different feature extractions are needed for each target value.

## Conclusion
As a solo team, I placed 29th out of 316 teams. Without NIR knowledge, it is hard to handle the spectral data. In fact, the biggest difference between higher rankers and me is feature engineering. However, the competition was very informative because it provided an opportunity to use many useful feature engineering such as interpolation, Fast Fourier Transform, and log scaling. Also, I conducted several models lncluding LightGBM, 1dCNN, Random forest, Xgboost and MLP at the beginning of the competition, hence, it was helpful to broaden my perspective on building diverse models.

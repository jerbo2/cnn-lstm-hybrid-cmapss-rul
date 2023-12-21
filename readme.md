### Rowan University ECE 09455 Machine Learning, Final Project

CNN-LSTM hybrid torch model for multivariate time series analysis, trained on [NASA's C-MAPSS dataset(s)](https://ntrs.nasa.gov/api/citations/20150007677/downloads/20150007677.pdf) comprised of aircraft engine run-to-failure data. This model leverages the spatial feature extraction capabilities of CNNs and the temporal sequence learning strengths of LSTMs to predict engine degradation over time. The aim is to provide accurate predictions for RUL (Remaining Useful Life) for a test unit given multi-sensor data. 

See the [report]() for more info on the approach, related works, and results. To replicate the results found in the report, explore the [notebook]() or make use of the [training]() and [evaluation]() scripts. Example for script use shown below.

```
# standard venv creation
python3 -m venv ./{VENV NAME}
source {VENV NAME}/bin/activate
pip install -r requirements.txt
```

To train the model, take a look at the training script mentioned above:

```
python3 cnn-lstm-hybrid-cmapss-training.py -h
```

Alternatively, load in the weights from ./best_saved_models using the evaluation script:

```
python3 cnn-lstm-hybrid-cmapss-eval.py -h
```
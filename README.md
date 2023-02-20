# Counterfactual Fairness through Data Preprocessing
We implement the Fair Learning through dAta Preprocessing (FLAP) algorithm to learn counterfactual fair decision through data preprocessing and the Conditional Distance Correlation test (Wang et al., 2015) to detect discrimination.

## Requirement
python >= 3.7


## Usage

### Fair learning
Admission example (Wang et al., 2019) with sample size 5000:
First run `main.py` with the data config file to generate the evaluation metrics
```
python ./main.py -c ./config/admission_n5000_eval_AllDiffScoreChange.json
```
Then run `plot.py` with the same config file to generate charts with previously generated metrics
```
python ./plot.py -c ./config/admission_n5000_eval_AllDiffScoreChange.json
```

### Fairness test
Admission example (Wang et al., 2019):
First run `main.py` with config files for different sample sizes
```
python ./main.py -c ./config/admission_n50_test_AllChange.json
python ./main.py -c ./config/admission_n100_test_AllChange.json
python ./main.py -c ./config/admission_n200_test_AllChange.json
```
Then run `plot.py` with the config file for the biggest sample size to plot the results for all sample sizes
```
python ./plot.py -c ./config/admission_n200_test_AllChange.json
```

Config files for other examples can be found in the `config/` folder. The configs for fairness learning evaluation have `_eval_` in their names, the configs for fairness test have `_test_` in their names.

### Real data analyses
We provide ipython notebooks for real data analyses

- Adult income data:
    the data is available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/), the analysis in `real-data-adult.ipynb` can be reproduced after downloading the `adult.data` and `adult.test` files and storing them under `data/Adult/` folder.
- COMPAS recidivism data:
    the data is available at the [ProPublica Repository](https://github.com/propublica/compas-analysis), the analysis in `real-data-compas.ipynb` can be reproduced after downloading the `compas-scores-two-years.csv` file and storing it under `data/COMPAS/` folder.
- Fintech data:
    the analysis is in `real-data-fintech.ipynb`.

# Counterfactual Fairness through Data Preprocessing
We implement the Fair Learning through dAta Preprocessing (FLAP) algorithm to learn counterfactual fair decision through data preprocessing and the Conditional Distance Correlation test (Wang et al., 2015) to detect discrimination.

## Requirement
python >= 3.7


## Usage

### Fair learning
Admission example (Wang et al., 2019) with sample size 5000
```
./main.py -c ./config/admission_n5000_eval_AllDiffScoreChange.json
```

### Fairness test
Admission example (Wang et al., 2019) with sample size 50
```
./main.py -c ./config/admission_n50_test_AllChange.json
```



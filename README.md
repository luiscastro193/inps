# INPS: Inference from Non-Probability Samples

Python package for statistical inference from non-probability samples.

## User guide

### Installation

INPS is available at the Python Package Index (PyPI).

```bash
pip install inps
```

### Running the examples

In order to run the code included in this guide, the following imports are required.

```python
import inps
import pandas as pd
import numpy as np
from numpy.random import default_rng
from xgboost import XGBRegressor, XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPRegressor
```

Also, the following code creates some simulated data as assumed in the examples.

```python
rng = default_rng(0)
pop_size = 10000
n = 1000
N = 2000
np_sample = rng.standard_normal(n * 5).reshape(-1, 5)
p_sample = rng.standard_normal(N * 3).reshape(-1, 3)
population = rng.standard_normal(pop_size * 3).reshape(-1, 3)
weights = [pop_size / N * 0.8] * int(N/2) + [pop_size / N * 1.2] * int(N/2)

def to_category(num_series):
	return pd.Series(np.where(num_series > 0, "Yes", "No"), dtype = "category", copy = False)

np_sample = pd.DataFrame(np_sample, columns = ["A", "B", "cat", "target", "target_cat"], copy = False)
p_sample = pd.DataFrame(p_sample, columns = ["A", "B", "cat"], copy = False)
population = pd.DataFrame(population, columns = ["A", "B", "cat"], copy = False)
np_sample["target_cat"] = to_category(np_sample["target_cat"])
np_sample["cat"] = to_category(np_sample["cat"])
p_sample["cat"] = to_category(p_sample["cat"])
population["cat"] = to_category(population["cat"])
p_sample["weights"] = weights
```

In general, `np_sample` and `p_sample` must be Pandas DataFrames. In a real example, the user has to verify the covariates. This implies making sure they have the same names in both DataFrame, same data types and same categories (if categorical).

All the code may be found in the [test script](test.py). Also, check the [guide script](guide.py) for code applying INPS to real data.

### Calibration

Calibration requires a sample and its known population totals. `population_totals` must be a Pandas Series.
```python
population_totals = pd.Series({"A": 10, "B": 5})
```
Additionally, the user must pass either the total population size...
```python
calibration_weights = inps.calibration_weights(np_sample, population_totals, population_size = pop_size)
```
...or the initial weights column name.
```python
calibration_weights2 = inps.calibration_weights(p_sample, population_totals, weights_column = "weights")
```
The resulting weights are a numpy array which may be used for estimation as usual.
```python
mean_estimation = np.average(np_sample["target"], weights = calibration_weights)
proportion_estimation = np.average(np_sample["target_cat"] == "Yes", weights = calibration_weights)
```

### Propensity Score Adjustment

PSA requires `np_sample`, `p_sample` and `population_size`.
```python
psa_weights = inps.psa_weights(np_sample, p_sample, pop_size)
```
The user may also pass a weights column for the `p_sample`.
```python
psa_weights = inps.psa_weights(np_sample, p_sample, pop_size, weights_column = "weights")
```
By default, columns with the same name will be selected as covariates. This may be dangerous. It is preferable to manually select the covariates after having verified them.
```python
psa_weights = inps.psa_weights(np_sample, p_sample, pop_size, weights_column = "weights", covariates = ["A", "B", "cat"])
```
By default, regularized logistic regression is applied. However, the user may choose any model supporting sample weights and compatible with the scikit-learn API.
```python
psa_weights2 = inps.psa_weights(np_sample, p_sample, pop_size, weights_column = "weights", model = XGBClassifier(enable_categorical = True, tree_method = "hist"))
```
For models requiring only numerical data without missing values, `make_preprocess_estimator` adds some default preprocessing.
```python
psa_weights3 = inps.psa_weights(np_sample, p_sample, pop_size, weights_column = "weights", model = inps.make_preprocess_estimator(BernoulliNB()))
```
The result is a dictionary with the `np_sample` and `p_sample` PSA weights as numpy arrays. The weights for the `np_sample` may be used for estimation as usual.
```python
mean_estimation = np.average(np_sample["target"], weights = psa_weights["np"])
proportion_estimation = np.average(np_sample["target_cat"] == "Yes", weights = psa_weights["np"])
```

### Statistical Matching

Matching requires `np_sample`, `p_sample` and `target_column` (from `np_sample`).
```python
mean_estimation = np.average(np_sample["target"], weights = psa_weights["np"])
proportion_estimation = np.average(np_sample["target_cat"] == "Yes", weights = psa_weights["np"])
```
It the target variable is categorical, a target category is required and probabilities are returned.
```python
cat_matching_values = inps.matching_values(np_sample, p_sample, "target_cat", "Yes")
```
By default, columns with the same name will be selected as covariates. This may be dangerous. It is preferable to manually select the covariates after having verified them.
```python
matching_values = inps.matching_values(np_sample, p_sample, "target", covariates = ["A", "B", "cat"])
```
By default, ridge regression (or regularized logistic regression for categorical values) is applied. However, the user may choose any model compatible with the scikit-learn API.
```python
matching_values2 = inps.matching_values(np_sample, p_sample, "target", model = XGBRegressor(enable_categorical = True, tree_method = "hist"))
```
For models requiring only numerical data without missing values, `make_preprocess_estimator` adds some default preprocessing.
```python
matching_values3 = inps.matching_values(np_sample, p_sample, "target", model = inps.make_preprocess_estimator(MLPRegressor()))
```
The result is a dictionary with the `p_sample` and `np_sample` imputed values (or probabilities if categorical) as numpy arrays. The values for the `p_sample` may be used for estimation as usual.
```python
mean_estimation = np.average(matching_values["p"], weights = p_sample["weights"])
proportion_estimation = np.average(cat_matching_values["p"], weights = p_sample["weights"])
```

### Doubly robust

The parameters are analogous to `matching`.
```python
doubly_robust_estimation = inps.doubly_robust_estimation(np_sample, p_sample, "target", covariates = ["A", "B", "cat"])
cat_doubly_robust_estimation = inps.doubly_robust_estimation(np_sample, p_sample, "target_cat", "Yes", covariates = ["A", "B", "cat"])
```
Default models are aplied. As usual, custom ones may be specified.
```python
doubly_robust_estimation2 = inps.doubly_robust_estimation(np_sample, p_sample, "target", psa_model = XGBClassifier(enable_categorical = True, tree_method = "hist"), matching_model = XGBRegressor(enable_categorical = True, tree_method = "hist"))
```
The estimated mean/proportion is directly returned by the method.

### Training

Training is the recommended method. The parameters and returning values are analogous to `matching`, except now there are 2 models the user may optionally specify.
```python
training_values = inps.training_values(np_sample, p_sample, "target", psa_model = XGBClassifier(enable_categorical = True, tree_method = "hist"), matching_model = XGBRegressor(enable_categorical = True, tree_method = "hist"))
```
The imputed values for the `p_sample` may be used for estimation as usual.

### Kernel weighting

Kernel weighting parameters are analogous to `psa`.
```python
kw_weights = inps.kw_weights(np_sample, p_sample, pop_size, weights_column = "weights", covariates = ["A", "B", "cat"])
```
A numpy array with the estimated weights for the `np_sample` is returned.
```python
proportion_estimation = np.average(np_sample["target_cat"] == "Yes", weights = kw_weights)
```

### Working with census data

The exact same methods can be used when the "probabilistic sample" includes the whole population instead.
```python
imputed_values = inps.training_values(np_sample, population, "target")
cat_imputed_values = inps.training_values(np_sample, population, "target_cat", "Yes")
mean_estimation = np.average(imputed_values["p"])
proportion_estimation = np.average(cat_imputed_values["p"])
```

### Advanced models

`inps.boosting_classifier()` and `inps.boosting_regressor()` will return advanced Gradient Boosting estimators ready to use for optimal results.
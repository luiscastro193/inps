import inps
import pandas as pd
import numpy as np
from numpy.random import default_rng
from xgboost import XGBRegressor, XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPRegressor

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

print(np_sample)
print(p_sample)

population_totals = pd.Series({"A": 10, "B": 5})
print(population_totals)
calibration_weights = inps.calibration_weights(np_sample, population_totals, population_size = pop_size)
print(calibration_weights)
calibration_weights2 = inps.calibration_weights(p_sample, population_totals, weights_column = "weights")
print(calibration_weights2)

mean_estimation = inps.estimation(np_sample["target"], calibration_weights)
mean_interval = inps.confidence_interval(np_sample["target"], calibration_weights)
print(mean_estimation)
print(mean_interval)
proportion_estimation = inps.estimation(np_sample["target_cat"] == "Yes", calibration_weights)
proportion_interval = inps.confidence_interval(np_sample["target_cat"] == "Yes", calibration_weights)
print(proportion_estimation)
print(proportion_interval)

psa_weights = inps.psa_weights(np_sample, p_sample, pop_size)
psa_weights = inps.psa_weights(np_sample, p_sample, pop_size, weights_column = "weights")
psa_weights = inps.psa_weights(np_sample, p_sample, pop_size, weights_column = "weights", covariates = ["A", "B", "cat"])
psa_weights2 = inps.psa_weights(np_sample, p_sample, pop_size, weights_column = "weights", model = XGBClassifier(enable_categorical = True, tree_method = "hist"))
psa_weights3 = inps.psa_weights(np_sample, p_sample, pop_size, weights_column = "weights", model = inps.make_preprocess_estimator(BernoulliNB()))

print(psa_weights["np"])
print(psa_weights["p"])

mean_estimation = inps.estimation(np_sample["target"], psa_weights["np"])
mean_interval = inps.confidence_interval(np_sample["target"], psa_weights["np"])
print(mean_estimation)
print(mean_interval)
proportion_estimation = inps.estimation(np_sample["target_cat"] == "Yes", psa_weights["np"])
proportion_interval = inps.confidence_interval(np_sample["target_cat"] == "Yes", psa_weights["np"])
print(proportion_estimation)
print(proportion_interval)

matching_values = inps.matching_values(np_sample, p_sample, "target")
cat_matching_values = inps.matching_values(np_sample, p_sample, "target_cat", "Yes")
matching_values = inps.matching_values(np_sample, p_sample, "target", covariates = ["A", "B", "cat"])
matching_values2 = inps.matching_values(np_sample, p_sample, "target", model = XGBRegressor(enable_categorical = True, tree_method = "hist"))
matching_values3 = inps.matching_values(np_sample, p_sample, "target", model = inps.make_preprocess_estimator(MLPRegressor()))

print(matching_values["p"])
print(matching_values["np"])
print(cat_matching_values["p"])
print(cat_matching_values["np"])

mean_estimation = inps.estimation(matching_values["p"], p_sample["weights"])
mean_interval = inps.confidence_interval(matching_values["p"], p_sample["weights"])
print(mean_estimation)
print(mean_interval)
proportion_estimation = inps.estimation(cat_matching_values["p"], p_sample["weights"])
proportion_estimation = inps.confidence_interval(cat_matching_values["p"], p_sample["weights"])
print(proportion_estimation)
print(proportion_interval)

doubly_robust_estimation = inps.doubly_robust_estimation(np_sample, p_sample, "target", covariates = ["A", "B", "cat"])
cat_doubly_robust_estimation = inps.doubly_robust_estimation(np_sample, p_sample, "target_cat", "Yes", covariates = ["A", "B", "cat"])
doubly_robust_estimation2 = inps.doubly_robust_estimation(np_sample, p_sample, "target", psa_model = XGBClassifier(enable_categorical = True, tree_method = "hist"), matching_model = XGBRegressor(enable_categorical = True, tree_method = "hist"))

print(doubly_robust_estimation)
print(cat_doubly_robust_estimation)

training_values = inps.training_values(np_sample, p_sample, "target", psa_model = XGBClassifier(enable_categorical = True, tree_method = "hist"), matching_model = XGBRegressor(enable_categorical = True, tree_method = "hist"))
print(training_values["p"])

kw_weights = inps.kw_weights(np_sample, p_sample, pop_size, weights_column = "weights", covariates = ["A", "B", "cat"])
print(kw_weights)

proportion_estimation = inps.estimation(np_sample["target_cat"] == "Yes", kw_weights)
print(proportion_estimation)

print(population)
imputed_values = inps.training_values(np_sample, population, "target")
cat_imputed_values = inps.training_values(np_sample, population, "target_cat", "Yes")
mean_estimation = inps.estimation(imputed_values["p"])
mean_interval = inps.confidence_interval(imputed_values["p"])
print(mean_estimation)
print(mean_interval)
proportion_estimation = inps.estimation(cat_imputed_values["p"])
proportion_interval = inps.confidence_interval(cat_imputed_values["p"])
print(proportion_estimation)
print(proportion_interval)

def my_estimator(np_sample, p_sample):
	matching_values = inps.matching_values(np_sample, p_sample, "target")
	return inps.estimation(matching_values["p"], p_sample["weights"])

advanced_interval = inps.advanced_confidence_interval(np_sample, p_sample, my_estimator)
print(advanced_interval)

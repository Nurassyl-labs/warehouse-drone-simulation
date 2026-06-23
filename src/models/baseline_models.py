from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

def get_baseline_models(random_state=42):
    """
    Returns a dictionary of baseline pose regressor models to evaluate.
    """
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
        "AdaBoost": MultiOutputRegressor(AdaBoostRegressor(n_estimators=100, random_state=random_state)),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=random_state, early_stopping=True)
    }

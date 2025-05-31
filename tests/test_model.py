def test_model_predicts_valid_class():
    import joblib
    import pandas as pd
    model = joblib.load("model/model.pkl")
    sample = pd.DataFrame([[2018, "BMW 3"]], columns=["Year", "Model"])
    pred = model.predict(sample)
    assert pred[0] in ["Good", "Bad"]

def test_model_has_predict():
    import joblib
    model = joblib.load("model/model.pkl")
    assert hasattr(model, "predict")

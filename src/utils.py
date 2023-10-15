from joblib import dump, load


class LoadClassifierThreshold:
    def __init__(self, model_path, threshold_path):
        try:
            self.model = load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")

        try:
            with open(threshold_path, "r") as threshold_file:
                self.threshold = float(threshold_file.read())
        except Exception as e:
            raise ValueError(f"Failed to load threshold from {threshold_path}: {str(e)}")

    def predict_with_threshold(self, testset):
        if not hasattr(self, 'model') or not hasattr(self, 'threshold'):
            raise ValueError("Model or threshold not loaded correctly.")

        try:
            # Use the predicted probabilities and compare with the threshold
            predicted_probabilities = self.model.predict_proba(testset)[:, 1]
            predictions = (predicted_probabilities >= self.threshold).astype(int)
            return predictions
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

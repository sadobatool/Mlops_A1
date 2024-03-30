import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TestHeartDiseasePrediction(unittest.TestCase):
    def setUp(self):
        # Load the data and split into train and test sets
        data = pd.read_csv('heart.csv')
        X = data.drop('target', axis=1)
        y = data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

        # Train the model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(self.X_train_scaled, self.y_train)

    def test_accuracy(self):
        # Test the accuracy of the model
        y_pred = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertAlmostEqual(accuracy, 0.79, places=0)  # Adjust expected accuracy as needed

    def test_classification_report(self):
        # Test the classification report
        y_pred = self.model.predict(self.X_test_scaled)
        report = classification_report(self.y_test, y_pred)
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1-score', report)

    def test_confusion_matrix(self):
        # Test the confusion matrix
        y_pred = self.model.predict(self.X_test_scaled)
        matrix = confusion_matrix(self.y_test, y_pred)
        self.assertEqual(matrix.shape, (2, 2))  # Ensure it's a 2x2 matrix

if __name__ == '__main__':
    unittest.main()

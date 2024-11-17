import unittest
import pandas as pd
from src.preprocess import preprocess_data
from src.rulefit import RuleFit

class TestPipeline(unittest.TestCase):

    def test_preprocess_data(self):
        # Test if data is loaded and preprocessed correctly
        file_path = 'data/sample_data.csv'
        X, y = preprocess_data(file_path, 'Class')
        
        # Check if X and y are not None
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

        # Check if the data has the expected shape
        self.assertEqual(X.shape[1], 2)  # Should have 2 features
        self.assertEqual(len(y), 5)      # Should have 5 rows in target

    def test_model_training(self):
        # Load the preprocessed data
        file_path = 'data/sample_data.csv'
        X, y = preprocess_data(file_path, 'Class')

        # Initialize and train RuleFit model
        model = RuleFit(tree_size=4, max_rules=2000, rfmode='classify',
                        model_type='rl', random_state=1, max_iter=1000)
        model.fit(X, y, feature_names=X.columns.tolist())

        # Check if the model has been fitted
        self.assertTrue(hasattr(model, 'rules_'))
        self.assertGreater(len(model.rules_), 0)

    def test_extract_rules(self):
        # Load the preprocessed data
        file_path = 'data/sample_data.csv'
        X, y = preprocess_data(file_path, 'Class')

        # Train the model
        model = RuleFit(tree_size=4, max_rules=2000, rfmode='classify',
                        model_type='rl', random_state=1, max_iter=1000)
        model.fit(X, y, feature_names=X.columns.tolist())

        # Extract rules and check if they are not empty
        rules_df = model.get_rules()
        self.assertIsInstance(rules_df, pd.DataFrame)
        self.assertGreater(len(rules_df), 0)

        # Check if the DataFrame contains expected columns
        self.assertIn('rule', rules_df.columns)
        self.assertIn('coefficient', rules_df.columns)
        self.assertIn('support', rules_df.columns)

if __name__ == '__main__':
    unittest.main()

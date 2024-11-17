import unittest
import pandas as pd
from src.preprocessing import preprocess_data
from rulefit import RuleFit

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Prepare test setup by loading data and preprocessing."""
        self.data_path = "data/input.csv"  # Adjust to match the actual path in your project
        self.target_column = "Class"

    def test_preprocess_data(self):
        """Test if data is preprocessed correctly."""
        try:
            X, y = preprocess_data(self.data_path, self.target_column)
            self.assertEqual(len(X), len(y))
            self.assertNotIn(self.target_column, X.columns)
            self.assertIsInstance(X, pd.DataFrame)
            self.assertIsInstance(y, pd.Series)
        except Exception as e:
            self.fail(f"Preprocessing failed with error: {e}")

class TestRuleFit(unittest.TestCase):
    def setUp(self):
        """Prepare test setup for RuleFit by preprocessing data."""
        self.data_path = "data/input.csv"
        self.target_column = "Class"
        X, y = preprocess_data(self.data_path, self.target_column)
        self.X = X
        self.y = y

    def test_rulefit_training(self):
        """Test if RuleFit model can train successfully."""
        rf = RuleFit(
            tree_size=4,
            max_rules=2000,
            rfmode='classify',
            model_type="rl",
            random_state=1,
            max_iter=1000
        )
        rf.fit(self.X, self.y)
        
        # Check if the RuleFit model has generated rules
        self.assertTrue(hasattr(rf, 'rules_'))
        self.assertGreater(len(rf.rules_), 0)

    def test_rule_extraction(self):
        """Test if RuleFit extracts rules correctly."""
        rf = RuleFit(
            tree_size=4,
            max_rules=2000,
            rfmode='classify',
            model_type="rl",
            random_state=1,
            max_iter=1000
        )
        rf.fit(self.X, self.y)
        
        # Extract the rules from the trained model
        rules = rf.get_rules()
        
        # Check if the rules are in the expected DataFrame format
        self.assertIsInstance(rules, pd.DataFrame)
        self.assertIn('support', rules.columns)  # Ensure the 'support' column exists
        self.assertIn('rule', rules.columns)    # Ensure the 'rule' column exists

    def test_rulefit_transform(self):
        """Test the transform method of the RuleFit model."""
        rf = RuleFit(
            tree_size=4,
            max_rules=2000,
            rfmode='classify',
            model_type="rl",
            random_state=1,
            max_iter=1000
        )
        rf.fit(self.X, self.y)
        
        # Transform the original data
        transformed_data = rf.transform(self.X)
        
        # Ensure that the transformed data has the expected shape
        self.assertEqual(transformed_data.shape[0], self.X.shape[0])  # Same number of rows
        self.assertGreater(transformed_data.shape[1], 0)  # Should have some columns (rules)

if __name__ == "__main__":
    unittest.main()

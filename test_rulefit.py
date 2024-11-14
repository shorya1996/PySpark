# tests/test_rulefit.py
import unittest
import pandas as pd
from preprocess import load_data, handle_missing_values, preprocess_data
from rulefit import RuleFit

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Load sample data for testing."""
        self.data = load_data("../data/Germancreditcardfraud.csv")
        self.target_column = 'Class'

    def test_load_data(self):
        """Test if data is loaded correctly."""
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertGreater(len(self.data), 0)

    def test_handle_missing_values(self):
        """Test handling missing values and column dropping."""
        cleaned_data = handle_missing_values(self.data, threshold=0.3)
        # Check if no columns have more than 30% missing values
        self.assertFalse(cleaned_data.isnull().mean().max() > 0.3)
        self.assertGreater(cleaned_data.shape[0], 0)

    def test_preprocess_data(self):
        """Test if preprocessing correctly splits features and target."""
        X, y = preprocess_data(self.data, self.target_column)
        self.assertEqual(len(X), len(y))
        self.assertNotIn(self.target_column, X.columns)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

class TestRuleFit(unittest.TestCase):
    def setUp(self):
        """Load sample data and preprocess for RuleFit testing."""
        data = load_data("../data/Germancreditcardfraud.csv")
        cleaned_data = handle_missing_values(data)
        X, y = preprocess_data(cleaned_data, 'Class')
        self.X = X
        self.y = y

    def test_rulefit_training(self):
        """Test if RuleFit model can train successfully."""
        rf = RuleFit(tree_size=4, max_rules=2000, rfmode='classify', model_type="rl", random_state=1, max_iter=1000)
        rf.fit(self.X, self.y)
        self.assertTrue(hasattr(rf, 'rules_'))
        self.assertGreater(len(rf.rules_), 0)

    def test_rule_extraction(self):
        """Test if RuleFit extracts rules correctly."""
        rf = RuleFit(tree_size=4, max_rules=2000, rfmode='classify', model_type="rl", random_state=1, max_iter=1000)
        rf.fit(self.X, self.y)
        rules = rf.get_rules()
        self.assertIsInstance(rules, pd.DataFrame)
        self.assertIn('support', rules.columns)
        self.assertIn('rule', rules.columns)

if __name__ == "__main__":
    unittest.main()

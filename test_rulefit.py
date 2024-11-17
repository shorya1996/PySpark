# test_rulefit.py
import unittest
import pandas as pd
from src.logger import get_logger
from src.preprocessing import preprocess_data
from src.model import RuleFit

# Initialize the logger
logger = get_logger(__name__)

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        """Load sample data for testing."""
        data_path = "data/input.csv"  # Update path if necessary
        target_col = 'Class'
        X, y = preprocess_data(data_path, target_col)
        self.X = X
        self.y = y

    def test_preprocess_data(self):
        """Test if preprocessing correctly splits features and target."""
        self.assertEqual(len(self.X), len(self.y))
        self.assertNotIn('Class', self.X.columns)
        self.assertIsInstance(self.X, pd.DataFrame)
        self.assertIsInstance(self.y, pd.Series)

class TestRuleFit(unittest.TestCase):
    def setUp(self):
        """Load sample data and preprocess for RuleFit testing."""
        data_path = "data/input.csv"  # Update path if necessary
        target_col = 'Class'
        X, y = preprocess_data(data_path, target_col)
        self.X = X
        self.y = y

    def test_rulefit_training(self):
        """Test if RuleFit model can train successfully."""
        try:
            rf = RuleFit(tree_size=4, max_rules=2000, rfmode='classify', model_type="rl", random_state=1, max_iter=1000)
            rf.fit(self.X, self.y)
            self.assertTrue(hasattr(rf, 'rules_'))
            self.assertGreater(len(rf.rules_), 0)
        except Exception as e:
            logger.error(f"Error during RuleFit model training: {e}", exc_info=True)
            self.fail(f"RuleFit training failed with exception: {e}")

    def test_rule_extraction(self):
        """Test if RuleFit extracts rules correctly."""
        try:
            rf = RuleFit(tree_size=4, max_rules=2000, rfmode='classify', model_type="rl", random_state=1, max_iter=1000)
            rf.fit(self.X, self.y)
            rules = rf.get_rules()
            self.assertIsInstance(rules, pd.DataFrame)
            self.assertIn('support', rules.columns)
            self.assertIn('rule', rules.columns)
        except Exception as e:
            logger.error(f"Error during Rule extraction: {e}", exc_info=True)
            self.fail(f"Rule extraction failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()

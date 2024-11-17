import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.main import main
from src.preprocessing import preprocess_data
from src.model import RuleFit

class TestRuleFit(unittest.TestCase):
    
    @patch("src.preprocessing.pd.read_csv")
    def test_data_preprocessing(self, mock_read_csv):
        """Test data preprocessing"""
        
        # Mock reading a CSV file
        mock_data = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6],
            'Class': [0, 1, 0]
        })
        mock_read_csv.return_value = mock_data
        
        # Call the preprocessing function
        data_path = "data/input.csv"
        target_col = "Class"
        X, y = preprocess_data(data_path, target_col)
        
        # Assertions
        self.assertEqual(X.shape, (3, 2))  # 3 rows, 2 features
        self.assertEqual(y.shape, (3,))  # 3 rows, 1 target column
        mock_read_csv.assert_called_with(data_path)
    
    @patch("src.model.RuleFit.fit")
    @patch("src.model.RuleFit.get_rules")
    def test_model_training(self, mock_get_rules, mock_fit):
        """Test model training"""
        
        # Mock the RuleFit model
        mock_rf = MagicMock(spec=RuleFit)
        mock_rf.get_rules.return_value = pd.DataFrame({
            'rule': ['Rule1', 'Rule2'],
            'coef': [0.5, -0.3],
            'support': [0.6, 0.4]
        })
        
        # Mock fit method
        mock_fit.return_value = mock_rf
        
        # Create dummy data
        X = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6]
        })
        y = pd.Series([0, 1, 0])
        
        # Train the model
        mock_rf.fit(X, y)
        
        # Assertions
        mock_fit.assert_called_once()  # Ensure fit is called
        rules_df = mock_get_rules.return_value
        self.assertEqual(rules_df.shape, (2, 3))  # 2 rules, 3 columns
    
    @patch("src.model.RuleFit.fit")
    @patch("src.model.RuleFit.get_rules")
    def test_main_function_error_handling(self, mock_get_rules, mock_fit):
        """Test error handling in the main function"""
        
        # Simulate error in model fitting
        mock_fit.side_effect = Exception("Model fitting failed")
        
        # Run the main function and expect it to handle errors
        try:
            main()  # Call the main pipeline
        except:
            pass  # Just pass the error to check if it fails gracefully

        # Ensure that fit was called, and we caught the error
        mock_fit.assert_called_once()

if __name__ == "__main__":
    unittest.main()

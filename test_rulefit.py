import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.main import main
from src.preprocessing import preprocess_data
from src.model import RuleFit

class TestRuleFit(unittest.TestCase):
    
    @patch("src.preprocessing.pd.read_csv")
    @patch("src.logger.get_logger")
    def test_data_preprocessing(self, mock_get_logger, mock_read_csv):
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Mock reading a CSV file
        mock_data = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6],
            'Class': [0, 1, 0]
        })
        mock_read_csv.return_value = mock_data
        
        # Run preprocessing function
        data_path = "data/input.csv"
        target_col = "Class"
        X, y = preprocess_data(data_path, target_col)
        
        # Assert that data is read correctly
        mock_read_csv.assert_called_with(data_path)
        self.assertEqual(X.shape, (3, 2))  # 3 rows, 2 features
        self.assertEqual(y.shape, (3,))  # 3 rows, 1 target column
        
        # Assert that logger info was called correctly
        mock_logger.info.assert_any_call("Starting data preprocessing...")
        mock_logger.info.assert_any_call("Data loaded from data/input.csv with shape (3, 3)")
    
    @patch("src.model.RuleFit.fit")
    @patch("src.model.RuleFit.get_rules")
    @patch("src.logger.get_logger")
    def test_model_training(self, mock_get_logger, mock_get_rules, mock_fit):
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Mock the RuleFit model
        mock_rf = MagicMock(spec=RuleFit)
        mock_rf.get_rules.return_value = pd.DataFrame({
            'rule': ['Rule1', 'Rule2'],
            'coef': [0.5, -0.3],
            'support': [0.6, 0.4]
        })
        
        # Mock fit method
        mock_fit.return_value = mock_rf
        
        # Run main function to test end-to-end process
        with patch("src.main.RuleFit", return_value=mock_rf):
            main()  # Call the main pipeline
            
        # Assert that RuleFit's fit method was called
        mock_fit.assert_called_once()
        
        # Assert get_rules method was called and returned a dataframe
        mock_get_rules.assert_called_once()
        rules_df = mock_get_rules.return_value
        self.assertEqual(rules_df.shape, (2, 3))  # 2 rules, 3 columns
        
        # Assert logger calls
        mock_logger.info.assert_any_call("Starting the end-to-end pipeline")
        mock_logger.info.assert_any_call("Model training completed successfully")
    
    @patch("src.model.RuleFit.fit")
    @patch("src.model.RuleFit.get_rules")
    @patch("src.logger.get_logger")
    def test_main_function_error_handling(self, mock_get_logger, mock_get_rules, mock_fit):
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Simulate error in model fitting
        mock_fit.side_effect = Exception("Model fitting failed")

        # Run main function and catch the exception
        with patch("src.main.RuleFit", return_value=MagicMock(spec=RuleFit)):
            main()

        # Assert logger error was logged
        mock_logger.error.assert_called_with("An error occurred: Model fitting failed", exc_info=True)

if __name__ == "__main__":
    unittest.main()

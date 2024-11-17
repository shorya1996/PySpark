import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.main import main
from src.model import RuleFit
from src.preprocessing import preprocess_data


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

    @patch("src.model.RuleFit")
    @patch("src.main.get_logger")
    def test_main_function_error_handling(self, mock_get_logger, MockRuleFit):
        """Test error handling in the main function"""

        # Mock the logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock for the RuleFit object
        mock_rf = MagicMock(spec=RuleFit)

        # Simulate that RuleFit is being called
        MockRuleFit.return_value = mock_rf

        # Add a print statement in the mock to track when fit is called
        def mock_fit(*args, **kwargs):
            print("fit method called!")
            return None
        mock_rf.fit = MagicMock(side_effect=mock_fit)  # Mock fit with side effect

        # Run the main function
        print("Running main function...")
        try:
            main()  # Call the main pipeline
        except Exception as e:
            print(f"Exception caught in main: {e}")  # Catch and print any error
        
        # Check if the fit method was called
        print(f"fit called: {mock_rf.fit.call_count}")  # Debug output for fit call
        
        # Assert that fit() was called once
        self.assertEqual(mock_rf.fit.call_count, 1, f"Expected 'fit' to be called once, but was called {mock_rf.fit.call_count} times")

        # Verify that the logger logged the error message
        mock_logger.error.assert_called_with("An error occurred: Model fitting failed")


if __name__ == "__main__":
    unittest.main()

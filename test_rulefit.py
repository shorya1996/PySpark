import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.main import main
from src.preprocessing import preprocess_data
from src.model import RuleFit
import logging


class TestPipeline(unittest.TestCase):

    @patch('src.preprocessing.pd.read_csv')  # Correctly patching the import path
    def test_data_preprocessing(self, mock_read_csv):
        # Sample data for mocking read_csv
        mock_data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [6, 7, 8, 9, 10],
            'Class': [0, 1, 0, 1, 0]
        })

        # Mock read_csv to return sample data
        mock_read_csv.return_value = mock_data

        # Test preprocess_data function
        data_path = "data/input.csv"
        target_col = "Class"
        X, y = preprocess_data(data_path, target_col)

        # Assertions to check if data is preprocessed correctly
        self.assertEqual(X.shape, (5, 2))
        self.assertEqual(y.shape, (5,))
        self.assertTrue('Feature1' in X.columns)
        self.assertTrue('Feature2' in X.columns)
        self.assertEqual(list(y), [0, 1, 0, 1, 0])

@patch('src.main.preprocess_data')  # Ensure correct path for preprocess_data
@patch('src.main.RuleFit')  # Mocking RuleFit where it's imported and used in main()
def test_model_training(self, mock_rulefit, mock_preprocess):
    # Sample processed data for testing
    mock_X = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
    mock_y = pd.Series([0, 1, 0])

    # Mock preprocess_data to return sample data
    mock_preprocess.return_value = (mock_X, mock_y)

    # Mock RuleFit instance
    mock_rf = MagicMock()
    mock_rulefit.return_value = mock_rf

    # Call the main function (the one we want to test)
    with patch('src.logger.get_logger') as mock_logger:
        mock_logger.return_value = logging.getLogger()

        main()  # Run the pipeline

        # Check if preprocess_data was called with correct arguments
        mock_preprocess.assert_called_once_with('data/input.csv', 'Class')

        # Check if RuleFit was initialized with the correct arguments
        mock_rulefit.assert_called_once_with(
            tree_size=4,
            max_rules=2000,
            rfmode='classify',
            model_type='rl',
            random_state=1,
            max_iter=1000
        )

        # Check if fit was called on RuleFit instance with the correct data
        mock_rf.fit.assert_called_once_with(mock_X, mock_y, feature_names=['Feature1', 'Feature2'])

        # Check if get_rules was called on RuleFit
        mock_rf.get_rules.assert_called_once()


    @patch('src.logger.get_logger')  # Mocking the logger
    @patch('src.model.RuleFit')  # Mocking RuleFit
    def test_pipeline_logging(self, mock_rulefit, mock_logger):
        # Mock logging
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance

        # Mock RuleFit behavior
        mock_rf = MagicMock()
        mock_rulefit.return_value = mock_rf

        # Run the main function
        main()

        # Ensure logger.info and logger.error are called appropriately
        mock_logger_instance.info.assert_any_call("Starting the end-to-end pipeline")
        mock_logger_instance.info.assert_any_call("Data preprocessing completed successfully")
        mock_logger_instance.info.assert_any_call("Model training completed successfully")

        # Check that no errors are logged if everything runs fine
        mock_logger_instance.error.assert_not_called()  # In this case, no error should occur


if __name__ == '__main__':
    unittest.main()

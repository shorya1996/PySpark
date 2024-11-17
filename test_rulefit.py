import unittest
from unittest.mock import patch, MagicMock
from main import main

class TestMain(unittest.TestCase):

    @patch("src.logger.get_logger")
    @patch("src.preprocessing.preprocess_data")
    @patch("src.model.RuleFit")
    def test_main_success(self, MockRuleFit, mock_preprocess_data, mock_get_logger):
        # Mocking logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mocking preprocess_data return value
        mock_preprocess_data.return_value = (MagicMock(), MagicMock())

        # Mock RuleFit
        mock_rf_instance = MagicMock()
        MockRuleFit.return_value = mock_rf_instance
        mock_rf_instance.fit.return_value = None
        mock_rf_instance.get_rules.return_value = MagicMock()

        with patch("builtins.print") as mock_print:
            main()

        # Assert the logger was called at the correct times
        mock_logger.info.assert_any_call("Starting the end-to-end pipeline")
        mock_logger.info.assert_any_call("Data preprocessing completed successfully")
        mock_logger.info.assert_any_call("Model training completed successfully")

        # Assert that print was called for the rules
        mock_print.assert_called_once()

    @patch("src.logger.get_logger")
    @patch("src.preprocessing.preprocess_data")
    @patch("src.model.RuleFit")
    def test_main_failure(self, MockRuleFit, mock_preprocess_data, mock_get_logger):
        # Mocking logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mocking preprocess_data to raise an exception
        mock_preprocess_data.side_effect = Exception("Error during preprocessing")

        with patch("builtins.print") as mock_print:
            main()

        # Assert the error log was called
        mock_logger.error.assert_any_call("An error occurred: Error during preprocessing")

        # Ensure that no model fitting or rule extraction happens
        MockRuleFit.assert_not_called()

if __name__ == "__main__":
    unittest.main()

@patch('src.rulefit.RuleFit')
@patch('src.preprocess.preprocess_data')
def test_model_training(self, mock_preprocess, mock_rulefit):
    # Sample processed data for testing
    mock_X = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
    mock_y = pd.Series([0, 1, 0])

    # Mock preprocess_data to return sample data
    mock_preprocess.return_value = (mock_X, mock_y)

    # Mock RuleFit instance
    mock_rf = MagicMock()
    mock_rulefit.return_value = mock_rf

    # Mock the behavior of get_rules().sort_values()
    mock_rules_df = pd.DataFrame({
        'rule': ['rule1', 'rule2'],
        'coefficient': [0.5, -0.2],
        'support': [100, 200],
        'importance': [0.8, 0.4]
    })
    mock_rf.get_rules.return_value = mock_rules_df
    mock_rf.get_rules.return_value.sort_values.return_value = mock_rules_df

    # Call the main function
    with patch('src.logger.get_logger') as mock_logger:
        mock_logger.return_value = logging.getLogger()
        main()

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

    # Check if get_rules was called
    mock_rf.get_rules.assert_called_once()

    # Check if sort_values was called on the DataFrame returned by get_rules
    mock_rf.get_rules.return_value.sort_values.assert_called_once()

# Soccer Match Predictor

A machine learning model to predict soccer match outcomes using historical data and advanced statistics.

## Features
- Team performance metrics
- Head-to-head statistics
- Form analysis
- Win streaks and clean sheets tracking
- SMOTE for handling class imbalance

## Requirements
```
pandas
numpy
scikit-learn
imbalanced-learn
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

```python
from soccer_predictor import SoccerPredictor

# Initialize predictor
predictor = SoccerPredictor()

# Load and prepare data
team_features = predictor.prepare_features(matches_df)
matches_df, feature_columns = predictor.create_matches_dataset(df, team_features)

# Train model
predictor.train(matches_df, feature_columns)

# Make prediction
test_features = {
    'home_goals_scored_rolling': 2.0,
    'home_goals_conceded_rolling': 1.0,
    'home_overall_form': 2.4,
    # ... other features ...
}

prediction, probability = predictor.predict_match(test_features)
```

## Model Features
- Goals scored/conceded (rolling average)
- Team form (overall and venue-specific)
- Clean sheet rate
- Scoring rate
- Win streaks
- Head-to-head statistics

## Performance
- Current accuracy: 51% (with synthetic data)
- Head-to-head statistics are strongest predictors
- Model performance improves with real historical data

## Future Improvements
- Add league positions
- Include player availability
- Consider weather conditions
- Add match importance metrics

## License
MIT

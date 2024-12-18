import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib


class SoccerPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        self.scaler = StandardScaler()

    def prepare_features(self, df):
        features_list = []

        for team in df['home_team'].unique():
            # Get all matches for form calculation
            team_matches = pd.concat([
                df[df['home_team'] == team],
                df[df['away_team'] == team]
            ]).sort_values('date')

            # Calculate points for form
            team_matches['points'] = np.where(
                (team_matches['home_team'] == team) & (team_matches['home_goals'] > team_matches['away_goals']) |
                (team_matches['away_team'] == team) & (team_matches['away_goals'] > team_matches['home_goals']),
                3,
                np.where(team_matches['home_goals'] == team_matches['away_goals'], 1, 0)
            )

            # Home stats
            home_matches = df[df['home_team'] == team].sort_values('date')
            home_form = team_matches['points'].rolling(window=5, min_periods=1).mean()

            home_stats = pd.DataFrame({
                'date': home_matches['date'],
                'team': team,
                'goals_scored_rolling': home_matches['home_goals'].rolling(window=5, min_periods=1).mean(),
                'goals_conceded_rolling': home_matches['away_goals'].rolling(window=5, min_periods=1).mean(),
                'overall_form': home_form,
                'venue_form': home_matches['home_goals'].rolling(window=5, min_periods=1).mean(),
                'clean_sheets': (home_matches['away_goals'] == 0).rolling(window=5, min_periods=1).mean(),
                'scoring_rate': (home_matches['home_goals'] > 0).rolling(window=5, min_periods=1).mean(),
                'win_streak': (home_matches['home_goals'] > home_matches['away_goals']).rolling(window=5,
                                                                                                min_periods=1).sum()
            })
            features_list.append(home_stats)

            # Away stats
            away_matches = df[df['away_team'] == team].sort_values('date')
            away_form = team_matches['points'].rolling(window=5, min_periods=1).mean()

            away_stats = pd.DataFrame({
                'date': away_matches['date'],
                'team': team,
                'goals_scored_rolling': away_matches['away_goals'].rolling(window=5, min_periods=1).mean(),
                'goals_conceded_rolling': away_matches['home_goals'].rolling(window=5, min_periods=1).mean(),
                'overall_form': away_form,
                'venue_form': away_matches['away_goals'].rolling(window=5, min_periods=1).mean(),
                'clean_sheets': (away_matches['home_goals'] == 0).rolling(window=5, min_periods=1).mean(),
                'scoring_rate': (away_matches['away_goals'] > 0).rolling(window=5, min_periods=1).mean(),
                'win_streak': (away_matches['away_goals'] > away_matches['home_goals']).rolling(window=5,
                                                                                                min_periods=1).sum()
            })
            features_list.append(away_stats)

        return pd.concat(features_list).sort_values('date').reset_index(drop=True)

    def add_h2h_features(self, matches_df):
        h2h_features = []

        for _, match in matches_df.iterrows():
            h2h_matches = matches_df[
                ((matches_df['home_team'] == match['home_team']) &
                 (matches_df['away_team'] == match['away_team']) |
                 (matches_df['home_team'] == match['away_team']) &
                 (matches_df['away_team'] == match['home_team'])) &
                (matches_df['date'] < match['date'])
                ]

            home_team_wins = len(h2h_matches[
                                     ((h2h_matches['home_team'] == match['home_team']) &
                                      (h2h_matches['home_goals'] > h2h_matches['away_goals'])) |
                                     ((h2h_matches['away_team'] == match['home_team']) &
                                      (h2h_matches['away_goals'] > h2h_matches['home_goals']))
                                     ])

            h2h_features.append({
                'date': match['date'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'h2h_matches': len(h2h_matches),
                'h2h_win_rate': home_team_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0.5,
                'h2h_goals_for': h2h_matches[h2h_matches['home_team'] == match['home_team']][
                    'home_goals'].mean() if len(h2h_matches) > 0 else 0,
                'h2h_goals_against': h2h_matches[h2h_matches['home_team'] == match['home_team']][
                    'away_goals'].mean() if len(h2h_matches) > 0 else 0
            })

        return pd.DataFrame(h2h_features)

    def create_matches_dataset(self, df, team_features):
        matches_df = df.copy()

        # Add team features
        for team_type in ['home', 'away']:
            for stat in ['goals_scored_rolling', 'goals_conceded_rolling', 'overall_form',
                         'venue_form', 'clean_sheets', 'scoring_rate', 'win_streak']:
                team_stats = team_features.copy()
                team_stats['date'] = team_stats['date'] + pd.Timedelta(days=1)

                matches_df = matches_df.merge(
                    team_stats[['date', 'team', stat]],
                    how='left',
                    left_on=['date', f'{team_type}_team'],
                    right_on=['date', 'team']
                )

                matches_df = matches_df.rename(columns={stat: f'{team_type}_{stat}'})
                matches_df = matches_df.drop(columns=['team'])

        # Add head-to-head features
        h2h_features = self.add_h2h_features(matches_df)
        matches_df = matches_df.merge(h2h_features, on=['date', 'home_team', 'away_team'])

        matches_df['target'] = (matches_df['home_goals'] > matches_df['away_goals']).astype(int)

        feature_columns = [
            'home_goals_scored_rolling', 'home_goals_conceded_rolling', 'home_overall_form',
            'home_venue_form', 'home_clean_sheets', 'home_scoring_rate', 'home_win_streak',
            'away_goals_scored_rolling', 'away_goals_conceded_rolling', 'away_overall_form',
            'away_venue_form', 'away_clean_sheets', 'away_scoring_rate', 'away_win_streak',
            'h2h_win_rate', 'h2h_goals_for', 'h2h_goals_against'
        ]

        return matches_df, feature_columns

    def train(self, matches_df, feature_columns):
        X = matches_df[feature_columns].fillna(0)
        y = matches_df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        X_train_scaled = self.scaler.fit_transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train_resampled)

        y_pred = self.model.predict(X_test_scaled)
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 5 Most Important Features:")
        print(importance.head())

        return X_test, y_test

    def predict_match(self, match_features):
        # Ensure features are in correct order
        features = np.array([
            match_features['home_goals_scored_rolling'],
            match_features['home_goals_conceded_rolling'],
            match_features['home_overall_form'],
            match_features['home_venue_form'],
            match_features['home_clean_sheets'],
            match_features['home_scoring_rate'],
            match_features['home_win_streak'],
            match_features['away_goals_scored_rolling'],
            match_features['away_goals_conceded_rolling'],
            match_features['away_overall_form'],
            match_features['away_venue_form'],
            match_features['away_clean_sheets'],
            match_features['away_scoring_rate'],
            match_features['away_win_streak'],
            match_features['h2h_win_rate'],
            match_features['h2h_goals_for'],
            match_features['h2h_goals_against']
        ]).reshape(1, -1)

        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        return prediction, probability

if __name__ == "__main__":
    # Create sample data
    matches = []
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'ManCity', 'ManUtd', 'Tottenham']
    start_date = '2023-01-01'

    np.random.seed(42)
    for i in range(500):
        home_team, away_team = np.random.choice(teams, 2, replace=False)
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)

        matches.append({
            'date': pd.to_datetime(start_date) + pd.Timedelta(days=i),
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals
        })

    df = pd.DataFrame(matches)

    predictor = SoccerPredictor()
    team_features = predictor.prepare_features(df)
    matches_df, feature_columns = predictor.create_matches_dataset(df, team_features)
    predictor.train(matches_df, feature_columns)

    # Test prediction
    test_features = {
        'home_goals_scored_rolling': 2.0,
        'home_goals_conceded_rolling': 1.0,
        'home_overall_form': 2.4,
        'home_venue_form': 2.2,
        'home_clean_sheets': 0.4,
        'home_scoring_rate': 0.8,
        'home_win_streak': 3,
        'away_goals_scored_rolling': 1.8,
        'away_goals_conceded_rolling': 1.2,
        'away_overall_form': 1.8,
        'away_venue_form': 1.5,
        'away_clean_sheets': 0.2,
        'away_scoring_rate': 0.6,
        'away_win_streak': 1,
        'h2h_win_rate': 0.6,
        'h2h_goals_for': 1.8,
        'h2h_goals_against': 1.2
    }

    prediction, probability = predictor.predict_match(test_features)
    print(f"\nMatch Prediction:")
    print(f"Outcome (1: home win, 0: draw/away): {prediction}")
    print(f"Win Probabilities [Away/Draw, Home]: {probability}")

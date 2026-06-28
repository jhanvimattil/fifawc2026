"""
FIFA World Cup 2026 - Round of 32 Match Predictions
Author: Professional Sports Analyst & Data Scientist
Date: June 28, 2026

This script implements an end-to-end Machine Learning pipeline to predict
the winners of the 16 matches in the Round of 32 of the 2026 FIFA World Cup.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
import os

def load_data():
    """Download and load results and shootouts data."""
    print("Step 1: Downloading dataset from repository...")
    results_url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    shootouts_url = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
    
    df = pd.read_csv(results_url)
    shootouts = pd.read_csv(shootouts_url)
    
    df['date'] = pd.to_datetime(df['date'])
    shootouts['date'] = pd.to_datetime(shootouts['date'])
    
    # Sort chronologically to maintain causality in simulation
    df = df.sort_values('date').reset_index(drop=True)
    return df, shootouts

def calculate_elo_and_features(df, shootouts):
    """
    Run chronological simulation to calculate dynamic Elo ratings,
    recent team form, and head-to-head records.
    """
    print("Step 2: Simulating match history to compute features...")
    
    # Initialize Elos
    elo_ratings = {}
    default_elo = 1500
    
    def get_elo(team):
        return elo_ratings.get(team, default_elo)
    
    # K-factor depends on the importance of the tournament
    def get_k_factor(tournament):
        t = str(tournament).lower()
        if 'fifa world cup' in t and 'qualification' not in t:
            return 60
        elif 'copa américa' in t or 'euro' in t or 'african cup of nations' in t or 'conmebol' in t:
            return 50
        elif 'qualification' in t or 'nations league' in t:
            return 40
        elif 'friendly' in t:
            return 20
        else:
            return 30

    elo_home_col = []
    elo_away_col = []
    
    # Form tracking: team -> list of match stats {'date': d, 'gd': gd, 'pts': pts}
    team_history = {}
    form_gd_home = []
    form_gd_away = []
    form_pts_home = []
    form_pts_away = []
    
    # Head-to-head tracking: (team1, team2) -> list of results from team1's perspective (1/0.5/0)
    h2h_records = {}
    h2h_home_rate = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        date = row['date']
        
        # Get Elo before match
        h_elo = get_elo(home)
        a_elo = get_elo(away)
        elo_home_col.append(h_elo)
        elo_away_col.append(a_elo)
        
        # Form helper
        def get_form_metrics(team, match_date):
            history = team_history.get(team, [])
            past = [h for h in history if h['date'] < match_date]
            if not past:
                return 0.0, 0.5  # GD = 0, Pts rate = 0.5 (neutral default)
            recent = past[-5:]
            avg_gd = np.mean([r['gd'] for r in recent])
            avg_pts = np.mean([r['pts'] for r in recent])
            return avg_gd, avg_pts

        hg, hp = get_form_metrics(home, date)
        ag, ap = get_form_metrics(away, date)
        form_gd_home.append(hg)
        form_gd_away.append(ag)
        form_pts_home.append(hp)
        form_pts_away.append(ap)
        
        # Head-to-head helper
        pair1 = (home, away)
        pair2 = (away, home)
        h2h_list = h2h_records.get(pair1, [])
        if len(h2h_list) > 0:
            h2h_h_win_rate = np.mean(h2h_list)
        else:
            h2h_rev = h2h_records.get(pair2, [])
            if len(h2h_rev) > 0:
                h2h_h_win_rate = 1.0 - np.mean(h2h_rev)
            else:
                h2h_h_win_rate = 0.5
        h2h_home_rate.append(h2h_h_win_rate)
        
        # If match is played, update ratings and history
        if not pd.isnull(row['home_score']) and not pd.isnull(row['away_score']):
            h_score = int(row['home_score'])
            a_score = int(row['away_score'])
            
            # Determine winner (resolving draws using shootout database)
            if h_score > a_score:
                winner = home
                w_h, w_a = 1.0, 0.0
            elif h_score < a_score:
                winner = away
                w_h, w_a = 0.0, 1.0
            else:
                match_shootout = shootouts[(shootouts['date'] == date) & 
                                           (((shootouts['home_team'] == home) & (shootouts['away_team'] == away)) |
                                            ((shootouts['home_team'] == away) & (shootouts['away_team'] == home)))]
                if len(match_shootout) > 0:
                    winner = match_shootout.iloc[0]['winner']
                    if winner == home:
                        w_h, w_a = 1.0, 0.0
                    else:
                        w_h, w_a = 0.0, 1.0
                else:
                    winner = None
                    w_h, w_a = 0.5, 0.5
            
            # Calculate expected score (standard Elo formula)
            dr = h_elo - a_elo
            expected_h = 1.0 / (1.0 + 10 ** (-dr / 400.0))
            expected_a = 1.0 - expected_h
            
            # Update ratings
            k = get_k_factor(row['tournament'])
            
            # Goal difference multiplier
            gd = abs(h_score - a_score)
            gdm = 1.0
            if gd == 2:
                gdm = 1.5
            elif gd == 3:
                gdm = 1.75
            elif gd >= 4:
                gdm = 1.75 + (gd - 3) / 8.0
                
            elo_ratings[home] = h_elo + k * gdm * (w_h - expected_h)
            elo_ratings[away] = a_elo + k * gdm * (w_a - expected_a)
            
            # Update history dictionaries
            if home not in team_history:
                team_history[home] = []
            if away not in team_history:
                team_history[away] = []
                
            pts_h = 1.0 if winner == home else (0.0 if winner == away else 0.5)
            pts_a = 1.0 if winner == away else (0.0 if winner == home else 0.5)
            
            team_history[home].append({'date': date, 'gd': h_score - a_score, 'pts': pts_h})
            team_history[away].append({'date': date, 'gd': a_score - h_score, 'pts': pts_a})
            
            # Update head to head
            if pair1 not in h2h_records:
                h2h_records[pair1] = []
            h2h_records[pair1].append(pts_h)

    # Assign features to main dataframe
    df['elo_home'] = elo_home_col
    df['elo_away'] = elo_away_col
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    df['form_gd_home'] = form_gd_home
    df['form_gd_away'] = form_gd_away
    df['form_pts_home'] = form_pts_home
    df['form_pts_away'] = form_pts_away
    df['h2h_home_rate'] = h2h_home_rate
    
    # Real home advantage indicator (takes host status into account)
    df['is_home_host'] = (df['country'] == df['home_team']).astype(int)
    
    return df

def build_model(df):
    """Train and evaluate the XGBoost Classifier."""
    # Target variable: 1 if home wins/advances, 0 if away wins/advances
    # Note: Drop pure draws from historical training set since knockout matches must have a winner
    def get_binary_target(row):
        if pd.isnull(row['home_score']) or pd.isnull(row['away_score']):
            return np.nan
        h = int(row['home_score'])
        a = int(row['away_score'])
        if h > a:
            return 1
        elif h < a:
            return 0
        else:
            # We don't have a shootout for this non-knockout draw, so return -1
            return -1

    df['target'] = df.apply(get_binary_target, axis=1)
    
    # We filter historical matches from 2010 onwards that are resolved (i.e. target is 0 or 1)
    df_played = df[(df['target'].notnull()) & (df['target'] != -1) & (df['date'] >= '2010-01-01')].copy()
    
    features = [
        'elo_diff', 'elo_home', 'elo_away',
        'form_gd_home', 'form_gd_away', 'form_pts_home', 'form_pts_away',
        'h2h_home_rate', 'neutral', 'is_home_host'
    ]
    
    # Time-based validation split: train before 2024, validate from 2024 to present
    train_mask = df_played['date'] < '2024-01-01'
    val_mask = df_played['date'] >= '2024-01-01'
    
    train_df = df_played[train_mask]
    val_df = df_played[val_mask]
    
    X_train, y_train = train_df[features], train_df['target']
    X_val, y_val = val_df[features], val_df['target']
    
    print(f"\nTraining set size: {len(train_df)} matches")
    print(f"Validation set size: {len(val_df)} matches")
    
    # Initialize XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        random_state=42,
        objective='binary:logistic'
    )
    
    # Train
    print("Training validation model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print("\n" + "="*50)
    print("VAL METRICS (Test set: matches 2024-2026)")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("="*50)
    print(classification_report(y_val, y_pred, target_names=['Away Wins/Advances', 'Home Wins/Advances']))
    
    # Retrain on all available historical data (train + validation) for final prediction
    print("\nRetraining model on complete historical dataset...")
    X_full = df_played[features]
    y_full = df_played['target']
    
    final_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        random_state=42,
        objective='binary:logistic'
    )
    final_model.fit(X_full, y_full)
    
    # Feature importances
    importances = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFinal Model Feature Importances:")
    print(importances)
    
    return final_model, features

def predict_upcoming_matches(df, model, features):
    """Run prediction on the 16 Round of 32 matches."""
    # Upcoming matches are rows where score is null and date is in World Cup 2026 knockout window
    df_future = df[df['home_score'].isnull() & (df['date'] >= '2026-06-28')].copy()
    
    print(f"\nStep 3: Predicting outcomes for {len(df_future)} Round of 32 matches...")
    
    # Extract feature values
    X_future = df_future[features]
    
    # Predict probabilities
    probs = model.predict_proba(X_future) # columns: [p_away_wins, p_home_wins]
    
    df_future['p_home_wins'] = probs[:, 1]
    df_future['p_away_wins'] = probs[:, 0]
    
    # Predict winner
    df_future['predicted_winner'] = np.where(
        df_future['p_home_wins'] >= 0.5,
        df_future['home_team'],
        df_future['away_team']
    )
    
    df_future['confidence'] = np.where(
        df_future['p_home_wins'] >= 0.5,
        df_future['p_home_wins'],
        df_future['p_away_wins']
    )
    
    # Formatting output table
    prediction_table = df_future[['date', 'home_team', 'away_team', 'p_home_wins', 'p_away_wins', 'predicted_winner', 'confidence']].copy()
    prediction_table['date'] = prediction_table['date'].dt.strftime('%Y-%m-%d')
    
    # Sort by date
    prediction_table = prediction_table.sort_values('date').reset_index(drop=True)
    
    return prediction_table

def main():
    df, shootouts = load_data()
    df = calculate_elo_and_features(df, shootouts)
    model, features = build_model(df)
    predictions = predict_upcoming_matches(df, model, features)
    
    print("\n" + "="*70)
    print("WORLD CUP 2026 ROUND OF 32 PREDICTIONS")
    print("="*70)
    print(predictions.to_string(index=False))
    print("="*70)
    
    # Save predictions as CSV
    predictions.to_csv("predictions_round_of_32.csv", index=False)
    print("Saved predictions to 'predictions_round_of_32.csv'.")

if __name__ == "__main__":
    main()

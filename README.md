professional machine learning pipeline for sports analytics, specifically predicting the winners of the 16 matches in the Round of 32 (Group 32) of the ongoing 2026 FIFA World Cup.

Goal
Build an end-to-end predictive model using historical international football match data from 1872 to June 28, 2026. The goal is to predict which teams will advance to the Round of 16, maximizing accuracy and F1 score.

Proposed Methodology
1. Data Ingestion
Download the comprehensive international football results dataset (results.csv) and penalty shootouts dataset (shootouts.csv) from the widely-used and frequently updated repository maintained by Mart Jürisoo (martj42/international_results).
2. Dynamic Rating System (Elo Ratings)
Rather than relying on static or outdated rankings, we will dynamically compute the historical Elo rating of every international team.

Initial Rating: All teams start with an Elo of 1500.
K-Factor Weighting:
FIFA World Cup Finals: $K = 60$
Continental Finals (Euro, Copa América, AFCON, etc.): $K = 50$
World Cup & Continental Qualifiers, Nations League: $K = 40$
Friendly Matches: $K = 20$
All other matches: $K = 30$
Margin of Victory (Goal Difference) Adjustment: Scale the rating change using a goal difference multiplier:
Goal Diff = 2: $1.5\times$ multiplier
Goal Diff = 3: $1.75\times$ multiplier
Goal Diff $\ge$ 4: $(1.75 + (GD - 3) / 8)\times$ multiplier
Draw Resolution: Match shootouts (shootouts.csv) will be used to resolve draws in tournament knockout stages, assigning a full win/loss to the respective teams for Elo calculation and target creation.
3. Feature Engineering
We will build predictive features for each match, calculated strictly chronologically to avoid data leakage:

elo_diff: Rating difference ($Elo_{home} - Elo_{away}$).
elo_home & elo_away: Absolute Elo ratings of both teams.
form_gd_home & form_gd_away: Average goal difference over the last 5 matches.
form_pts_home & form_pts_away: Average points earned (3 for win, 1 for draw, 0 for loss) over the last 5 matches.
h2h_home_rate: Historical head-to-head win/loss record between the two teams.
neutral: Binary flag indicating if the game is played on a neutral site.
is_home_host: Binary flag indicating if the home team is one of the host countries (USA, Canada, Mexico) and is playing in their home country, preserving the home advantage even for neutral-site tournament games.
4. Model Training & Evaluation
We will train a binary classification model (Home Team wins/advances vs Away Team wins/advances) on matches from 2010 onwards that did not end in pure draws (or had draws resolved by shootouts).

Model: XGBoost Classifier.
Validation Scheme: Time-based split. Train on matches from 2010 to 2023. Test/Validate on matches from 2024 to June 2026.
Metrics: Report Accuracy, F1 Score, Precision, and Recall.
Prediction Inference: Run predictions on the 16 Round of 32 matchups, outputting probabilities and predicted winners.
Proposed Changes
main
[NEW] 
predict_wc32.py
This script will contain the entire machine learning pipeline: downloading data, running historical simulations to calculate Elo and team form, feature engineering, model training/evaluation, and running predictions for the 2026 Round of 32.

[NEW] 
predictions_report.md
A detailed markdown report describing the results of our model, feature importances, and the exact predicted winners for all 16 matches of the Round of 32 with probabilities.

Verification Plan
Automated Verification
Run predict_wc32.py to train the model, evaluate metrics on the validation set, and output predictions.
Verify that accuracy is high (typically $>68%$ for binary outcome predictions) and F1 score is excellent.
Manual Verification
Review the output table in predictions_report.md to ensure matches like Brazil vs. Japan, Germany vs. Paraguay, etc. have clear predicted winners and probability distributions.

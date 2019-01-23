# nba-shaq
NBA Seasonal Honors &amp; Accolades Quantifier (SHAQ)

SHAQ is a machine learning model and Flask front-end for predicting winners
of the NBA's regular season awards, like Most Valuable Player (MVP), Most
Improved Player (MIP), etc.

SHAQ's predictions are based on per 100 possession statistics dating from 1980, when the 3 point line was first introduced to the NBA. Per 100 stats are used to account for differences in pace across eras. (Today's NBA game is about 10% slower than it was in 1980. That's right, slower!)

SHAQ's predictive model uses a soft voting average of xgboost plus scikit-learn's random forest classifiers.

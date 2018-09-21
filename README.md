# nba-shaq
NBA Seasonal Honors &amp; Accolades Quantifier (SHAQ)

SHAQ is a machine learning model and Flask front-end for predicting winners
of the NBA's regular season awards, like Most Valuable Player (MVP), Most
Improved Player (MIP), etc.

Right now, SHAQ provides bare-bones "predictions" for the 2017-18 MVP award (given to James Harden in June 2018). Once the 2019 NBA season starts, SHAQ should be able to give you up-to-date predictions for 2019 NBA accolades.

SHAQ's predictions are based on per 100 possession statistics dating from 1980, when the 3 point line was first introduced to the NBA. Per 100 stats are used to account for differences in pace across eras. (Today's NBA game is about 10% slower than it was in 1980. That's right, slower!)

SHAQ's predictive model uses a soft voting average of xgboost plus scikit-learn's AdaBoost and random forest classifiers.
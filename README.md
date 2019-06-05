# HOME
**HO**pefully-s**M**art n**E**ws aggregator

RSS Feed news aggregator + some machine learning magic to make it nice.

import db: mongorestore --db homedb dump/homedb

# TODO
- Change slide with performance from regular stratified cross-validation
- Add "Feature Selection" within TF-IDF (slide)
- Add "Independent Feature Selection" and show how it sucks (ADDITIONAL TESTS MAY BE REQUIRED!)
- Add "Paired T-Test" slide
- Show briefly how feature selection works with Bagging (which btw sucks) (slide)
- Remove Learning-Curves


# Slides
- Intro
- ...
- Base classifiers (NC)
    - Regular CV Scores *** 
    - Base Classifiers + Independent Features Selection (Regular CV) ***
- Ensemble/Meta Classifiers
    - Regular CV Scores ***
    - Just a little bit of Feature Selection within Meta-Classifiers ***
- Statistical Tests ***
    - Table with Paired-wise comparison
    - Final Ranking recap (whatever that means)
- Model Selection
- Application Demo


## Important stuff
- Attribute selection (only on test-set, inside a meta-classifier)

## Extra stuff
- Remove similar articles when building the model (?)
- Try predict likability without category ()
- KNN for suggestions 
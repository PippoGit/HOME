# HOME
**HO**pefully-s**M**art n**E**ws aggregator

RSS Feed news aggregator + some machine learning magic to make it nice.

import db: mongorestore --db homedb dump/homedb

# TODO

## Important stuff
- Statistical tests (t-paired)
- Attribute selection (only on test-set, inside a meta-classifier (??))
- Attribute selection + Meta Classifiers (?)
- Is plot_learning_curve ok?
- Eval AUC_ROC for all the classifiers!
- Repeat the Standard 10folds CrossValidation for all the classifiers
- Tests everything (classifiers + attribute selection)

## Extra stuff
- Remove similar articles when building the model (?)
- Try predict likability without category ()
- Subset evaluation
- KNN for suggestions 
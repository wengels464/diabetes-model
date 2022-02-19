# This is an exploration of the UCI Pima Indian Diabetes Dataset

## Open questions:

### Honesty of using new features that are defined manually in the dataset?

It seems like it's OK as long as the model is cross-validated with k-folds?

### Honesty of replacing NaN values with median (from label)?

If your dataset is incomplete is it considered OK to replace NaNs with median values? It seems as if the data is being manipulated?
Or is it more a case of, do anything as long as it can be cross-validated and improves the accuracy score?

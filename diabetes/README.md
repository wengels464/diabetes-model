# This is an exploration of the UCI Pima Indian Diabetes Dataset

## Characteristics

This contains 758 observations with 8 attributes. 

It contains 500 negative entries and 258 positive entries.

The dataset is sourced from female Pima Indians over the age of 21.

## Proposed Features and Outcomes

### Containerization

I would love to produce my first Docker container.

### Cloud Hosting

It appears that Amazon ECS (Fargate) supports Docker containers.

### Functionality

I need some help understanding how something like this can be turned into a production-worthy piece of code.

My understanding is this:

Static dataset is used to train an algorithm.

Algorithm is then put into a new program capable of taking in new data and making predictions.

In addition, new program is capable of adding new data to a database SQL-style.

Whole thing is put on the cloud (AWS, Redshift) with a web UI that enables new records to be uploaded, analyzed, and stored.

Finally the whole thing creates a dashboard that allows the user (let's say an HMO) to do population analysis and discern, for instance, whether or not rates of diabetes are increasing or decreasing in a particular population group.

There's also logs to consider as well.

It would seem that there is a bit of a problem here, as the new data being entered is unlabeled, receives a predicted label, and then most likely will be given a true label.

If this is the case, then it would seem that the addition of the true label should trigger a retraining of the algorithm.

Unsure how to make this work continuously, or if this sort of thing is even done.

### Overall

I'd like for the user to be able to upload new data either individually or in batches (CSV) and then have that data analyzed, appended to a cloud SQL server, and ultimately visualized.

I'm picturing someone in a hospital or other health org or even insurance company being able to track population metrics over time and visualize trends.

### Concerns

Lots of unknowns around cloud computing and web development. Some familiarity with HTML5 and CSS but unsure about Flask, Django, etc.

Unsure how to make these different tools (Python, SQL, Docker, AWS) "talk" to each other and ensure compatibility.

## Open questions:

### Honesty of using new features that are defined manually in the dataset?

It seems like it's OK as long as the model is cross-validated with k-folds?

### Honesty of imputing NaN values with median (from label)?

If your dataset is incomplete is it considered OK to replace NaNs with median values? It seems as if the data is being manipulated?
Or is it more a case of, do anything as long as it can be cross-validated and improves the accuracy score?

### Ensemble learning

Can you just do a grid search for every applicable algorithm and every applicable hyperparameter to get your ideal fit?

### Mapping algorithms to problems

Is there a resource or guide for going from: "I have problem of Type A, here are algorithms that can be used for Type A problems"? Classification v. regression, etc?

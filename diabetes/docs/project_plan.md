# Project Plan for Diabetes Dataset Project

## Key Outcomes:

1. Create an algorithm that is not overfit to the data but has low bias.
2. Create a SQL database from which the original data can be read.
3. Use that same SQL database to add new data.
4. Use the addition of new, labeled data to trigger a retraining event.
5. Analyze unlabeled data.
6. Containerize and deploy to the cloud with a web UI.

### Algorithm

Since the data in question is small and we're showing off, ensemble learning might be appropriate.

It may be worthwhile to have two training sets, one with new features suitable for non-neural algorithms and one without for Keras.

### SQL (Input and Output)

I will have to research a couple of things:
1. Cloud-based operation of SQL client and server.
2. Putting data into the server will be straightforward, but I'm unsure how SQL can send a signal BACK to Python indicating that it has been updated (this signal would trigger retraining)

### Flow of Data

Raw CSV 
Pandas Dataframe
Cleaning and Imputation
New Dataframe
Clean True CSV
Invent New False CSV
Algorithmic Training
Predict Labels on False CSV
Dashboard Software (PowerBI or something)
...

*What I would like is a way to pipe data into this program and have it analyze, label, and store the data in SQL, then have SQL talkk to a dashboard. I'm unsure of the details*

## Final Thoughts

This should be the sort of thing that you can show an insurance company and say, "Imagine this was real data. This algo would give you actionable insight and the backend built around it would allow you to add new patients, receive a prediction, and retrain as new true labels are applied".

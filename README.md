# This is an exploration of the UCI Pima Indian Diabetes Dataset

## Characteristics

This contains 758 observations with 8 attributes. 

It contains 500 negative entries and 258 positive entries.

The dataset is sourced from female Pima Indians over the age of 21.

## Proposed Features and Outcomes

### Containerization & Reproducibility

Product will containerized in a Docker container running a minimal Ubuntu/Conda install.

Python dependencies will be managed using Conda.

### Cloud Hosting

Deployable to ECS/ECR/Fargate, uses Flask API for user interaction and HTML styling.

Once spun up, provides an HTML form that allows the following user characteristics to be entered:
- Age
- Blood Sugar
- Insulin
- BMI
- Diabetes Pedigree Function


### Functionality

0. A docker container with dependencies pre-loaded is spun up on ECS/ECR.

1. Data is loaded into a pandas dataframe.

2. Missing data is imputed using MICE (multiple-imputation using chained equations)

3. Data is split into 80/20 training/testing subsets.

4. Various models are trained and optimized for recall. KNN, XGBoost, and Random Forest are explored. No hyperparameter tuning.

5. Optimal model is pickle'd to an output file.

6. A second script runs Flask with only the outputted model. 

7. User input is taken and a prediction made, which returns either likely for type 2 diabetes or unlikely.

### Use Cases

A health system or even insurance company will be able to track population metrics over time and visualize trends.

Individual patients can understand their risk profile and screen themselves for chronic conditions.

Population health managers at an HMO can flag patients likely to develop long-term complications and have them scheduled for screening (A1C).


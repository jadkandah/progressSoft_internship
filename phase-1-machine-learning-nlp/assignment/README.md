# assignment

## Objective
(.venv) jadkandah@trn757:~/progressSoft_internship$ python /home/jadkandah/progressSoft_internship/phase-1-machine-learning-nlp/assignment/src/client.py
Text: I absolutely love this new update!
Predicted sentiment: Positive
------------------------------------------------------------
Text: This game is terrible and completely broken.
Predicted sentiment: Negative
------------------------------------------------------------
Text: The maintenance update begins tomorrow.
Predicted sentiment: Neutral
------------------------------------------------------------
## Concepts

## Tasks

## Notes
- The dataset contained missing values exclusively in the text column. Since the tweet text is the primary feature used for sentiment classification, these rows could not be meaningfully imputed. Replacing the missing text with placeholders or empty strings would introduce artificial samples with no semantic information. Therefore, rows with missing tweet text were removed prior to preprocessing.

- Trained baseline logistic regression and linear svm models and saved their results with the better model being the linear svm baseline model.

- Hyperparameter tuning improved the performance of both classifiers. Logistic Regression benefited more from tuning, improving its Macro F1-score by approximately 0.8%. However, the tuned Linear SVM still achieved the best overall performance, reaching 98.0% accuracy and Macro F1-score, making it the final selected model for deployment.

## Resources

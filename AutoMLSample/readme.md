#### 1. Run AutoTrain (AutoML training) and return metrics (experimentResult)
#### 2. For each trainer (get trainers from experimentResult.RunDetails)
- 2.1 Run PFI to detect what features are not needed (we have to define rules for not needed)
- 2..2 Run Correlation Matrix to detect what features are redundant (we have to define rules for redundant)
- 2.3 Remove not needed or reduntant features and go to step 1.
- 2.4 If no more features to remove go to step 3.
#### 3. Return best trainer

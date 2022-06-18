# graphical-wellbeing-assessment

This repository is part of the [CSE3000 Research Project](https://github.com/TU-Delft-CSE/Research-Project) 2022 of [TU Delft](https://github.com/TU-Delft-CSE)

## Important Information
The pre-trained models that are used in the code for inference are too large to be pushed in this repository. To obtain the 9 models that the application uses for inference follow these steps:
In the following steps `"Element"` is to be replaced with either `"House", "Tree", "Person"` depending on the model you wish to obtain.

 1. Un-comment the lines 150-155 in `ElementClassifier.py` depending on the model you want to obtain.
 2. Adjsut the model parameters here
 `checkpoint_losses = training(model, device, 0.0001, num_epochs, train_loader)`
 3. Replace `num_epochs` with the number of desired epochs for training
 4. For each of the elements (House, Tree, Person) three separate models were trained with 10, 12, and 15 epochs. The learning rate was 0.0001 for every model.
 5. The snippet in lines 150-155 will store the model  for example: `house_model_12.tar'`. This represents a model from the house classifier trained with 12 epochs.
 6. Run `ElementClassifier.py`
 7. Repeat the steps for each of the elements to obtain all 9 models.

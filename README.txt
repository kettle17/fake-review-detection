PRETRAINED MODELS ARE MISSING FROM THIS SUBMISSION DUE TO FILESIZE
PLEASE DOWNLOAD THEM FROM ... AND PUT THEM IN THE "modelload" FOLDER 

========================
Initialisation
========================

By running main.py in CMD, the program will start.
It will first output if the GPU is available for use - outputting "cuda:0" if available.
If this is not the case then it will output a warning as CPU computation takes much longer than GPU computation during the training process.

========================
Dataset loading
========================

The program will then prompt the user for their choice of dataset:
1. FRDDS mode - The program will then check for the FRDDS dataset being present, and load it.
2. Custom mode - The program will take a name of a custom csv dataset in the dataset/custom folder and load it. It must be in the same general format as FRDDS.

The program will then prompt the user for train:eval:test splits:
1. FRDDS mode - For a dataset of 4000, this will be (2600:700):700, a 82.5%:17.5% test split considering eval is used for training. This is not the normal test split ratio due to the small dataset size.
2. Custom mode - The program will take 3 integer values representing percentage which MUST add up to 100. The sizes of each set will be output, and the user will be prompted again to confirm.

The program will then prompt the user for their choice in dataset mode:
1. Review text only (default)
2. Review title and text
3. Review title only

This will load the dataset in this mode, and all training and testing will also be done in this mode.

========================
Training
========================

The program will then prompt the user if they would want to use the best pretrained models, or if they want to train their own. Choosing the pretrained models will load the model associated with the dataset mode chosen earlier, and skip to testing. Choosing a custom model will instantiate a new RobertaForBinaryClassification model.

The program will then ask if the user wants to use existing best parameters for learning rate, optimizer parameters, etc, or if they would like to enter their own parameters. Once entered, the program will begin the training process. 
On completion of an epoch, the program will output the model accuracy on the eval set, and training and evaluation losses. 
On completion of model training, the program will output the best epoch's accuracy, and prompt to use it in testing.

========================
Testing
========================
If the user has skipped training, the program will redirect here.

The program will test the model on the test set, and display the testing accuracy along with other metrics.
The program will then hang until the user chooses to exit.


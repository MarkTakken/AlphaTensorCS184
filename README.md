## Info
Before running, data should be generated into a data/ folder, and, if using a pretrained model, that model should be placed into a models/ folder. 

## Download

To use the trained model mentioned in the paper, use model_32_4.pt, which is in results.

## Primary Code:

The following python files hold central classes and functions of AlphaTensorCS184:

architecture.py
training.py
selfplay.py
tensorgame.py
training.py
utilities.py

Additionally, main.py holds the three primary modes in which you may want to run the loop at-large.

Prior to running a training method, you should generate data with example_generation.
The samples of data are too large to store, and relatively fast to generate.

If you are running this program, the following additional folders are used:

models/
data/ 

Within tests/, there are three scripts used for testing. They should be put into the primary folder before running:

tests.py
testing.ipynb
selfplay_tests.py

analysis.ipynb is used to generate information for the report, based on information within results. 

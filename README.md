# Quantum-Black-Challenge

## Data 

To properly get the data, one has to download it locally on his/her computer, unzip and put the folder in the main folder. One should have a folder like data/ai_ready/ with 6 different elements:
- images/ : folder containing the images
- masks/ : folder containing the masks
- train_images/ : folder containing two empty subfolders (0 and 1)
- val_images/ : folder containing two empty subfolders (0 and 1)
- test_images : folder containing one subfolder named 'test'
- x-ai_data.csv : csv file containing classes for each image.

## Repository description

### model_building
Folder containing the python files to create the different models.

The most efficient model is stored in new_keras_model.py.

### model_test
Folder containing the python file to test the model and make prediction.

### app
Folder containing the file to create a web app that integrates the model and its predictions.

**Instructions to create the app:**

- Create a new environment:

```pipenv shell```

- Install requirements:

```pip install -r requirements.txt```

- Launch the app:

```streamlit run Home.py```

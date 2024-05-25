# AGE AND GENDER CLASSIFICATION USING CNN

## Project Overview

The classification of age and gender has drawn increased attention recently because
of its significance in creating user-friendly intelligent systems. Age estimation from a
single facial image has been a key task in the fields of image processing and computer
vision. Convolutional Neural Network (CNN) based techniques have been frequently
adopted for the classification problem in the recent past because of their precise results in
facial analysis. This study presents an end-to-end CNN approach for obtaining accurate
gender and age group classification of real-world faces. The complete feature extraction
and classification processes are included in the two-level CNN architecture. The feature
extraction task pulls features that are related to gender and age while the classification
assigns the facial photographs to the proper gender and age group. The experiment
results appear to support the claim that our model may perform better in gender and
age group categorization when analysed for classification accuracy using the equivalent
Adience benchmark. Technically speaking, our network will be trained and tested on
both Adience (original) and IMDB-WIKI datasets.

## Repository Structure

```plaintext
Age_and_Gender_Classification/
│
├── 1_Gender classification and Face-final.ipynb
├── 2_Age classification and Face-final.ipynb
├── 3_Pipeline.ipynb
├── 4_Pipeline-Age.ipynb
├── 5_cleaning and preprocessing imdbwiki.py
├── documents/
│   └── [files include project_report and log_book]
├── Instructions.txt
├── model and encoders/
│   └── [model and encoder files]
├── Paper Publication/
│   └── [files include published paper, pulication link, conference certificate]
├── reference/
│   └── [files include all reference notebooks and python scripts]
├── shape_predictor_68_face_landmarks.dat
└── Video Demonstration/
    └── [files include Architecure diagram and demonstration video]
    
```

## Instructions
To set up and run the project, follow these steps:

- Download the required dataset online.
- Create a virtual environment for Python and install all the required packages.
- Run the provided Jupyter notebooks to extract and analyze the data.

## Environment Setup
To create a virtual environment and install the necessary packages, execute the following commands:

```bash 
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

## Data

In your project directory create a folder named **data** and load the datasets here

The datasets **Adiance** and **IMDB-WIKI** are invaluable assets for the field of face detection and prediction.

> **URL:** [Adiance Dataset](https://www.kaggle.com/datasets/alfredhhw/adiencegender/data) and [IMDB-WIKI Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- **Content:** Roughly 524,000 high-resolution RGB images.

Unzip both files in the same locations.

## Running the Notebooks

### Activate the virtual environment:

```bash 
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### Launch Jupyter Notebook:

```bash 
jupyter-notebook
```

### Open and execute the notebooks:

- 1_Gender classification and Face-final.ipynb
- 2_Age classification and Face-final.ipynb
- 3_Pipeline.ipynb
- 4_Pipeline-Age.ipynb
- 5_cleaning and preprocessing imdbwiki.py

## Project Demonstration
You can find the video demonstration of the project in the Video Demonstration folder. [Click here to watch it](images%20and%20videos/Video_Demonstration.mp4).

## Architecture Diagram

### System Architecture
![System Architecture](images%20and%20video/system%20architecture.png)

### CNN Architecture
![CNN Architecture](images%20and%20video/CNN%20architecture.png)

## Additional Documentation
Further details about the project can be found in the documents folder. This includes the project report, references, and related publications.



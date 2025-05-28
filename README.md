# DT_Crystallography

**DT_Crystallography** is a Python-based project focused on data processing and analysis in crystallography. It includes scripts for data loading, transformation, cropping, merging along with machine learning models for predicting phases based on amplitudes.

## Project Structure
DT_Crystallography/

├── analyzeData_all.py     # Script for running loadData_all.py that process specified cif files

├── loadData_all.py        # Loads and preprocessesing cif files into NumPy ndarrays

├── merged_file.py         # Merges multiple NumPy ndarrays

├── crop_data.py           # Crops datasets based on specified mode(full, half, half+)

├── models_loss.py         # Defines ML models and loss functions

├── transform_load_data.py # Defines DataLoader and transformotaion of train and validation dataset

├── train_model.py         # Trains model and predict phases.

├── README.md              # Project documentation

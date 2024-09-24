Graduate Admission Prediction Using Artificial Neural Networks (ANN)

This repository contains a project focused on predicting the likelihood of graduate school admission using an Artificial Neural Network (ANN). The dataset used in this project includes various features such as GRE score, TOEFL score, university rating, and more, which are crucial for determining a student's admission chances.

🗂 Project Overview

The goal of this project is to build a predictive model that estimates the probability of a student being admitted to a graduate program based on several academic and profile features. This project demonstrates the application of deep learning techniques, specifically an ANN, for binary classification.

📊 Dataset

The dataset used in this project contains the following features:
- GRE Score: Graduate Record Examination score.
- TOEFL Score: Test of English as a Foreign Language score.
- University Rating: Rating of the university (1 to 5).
- SOP: Statement of Purpose strength (1 to 5).
- LOR: Letter of Recommendation strength (1 to 5).
- CGPA: Undergraduate GPA.
- Research: Whether the student has research experience (0 or 1).
- Chance of Admit: Probability of admission (target variable).

🧠 Model Architecture

The ANN model is built using the following architecture:
- Input Layer: Receives input features.
- Hidden Layers: Multiple layers with ReLU activation functions to capture complex patterns.
- Output Layers: Single neuron with a sigmoid activation function to predict the probability of admission.

Model Details:
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam
- Metrics: Accuracy

🛠 Prerequisites

To run this notebook, you need to have the following libraries installed:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow/Keras
- Scikit-learn

🖥 Usage

1. Data Preparation: The notebook includes steps for loading and preprocessing the data.
2. Model Training: You can train the model by running the respective cells in the notebook.
3. Evaluation: The trained model is evaluated on the test set, and the results are displayed with appropriate visualizations.

🔍 Future Work

Some potential improvements for this project include:
- Exploring other machine learning models like Random Forest or SVM.
- Hyperparameter tuning using Grid Search or Bayesian Optimization.
- Incorporating additional features such as demographic data for a more comprehensive model.

📫 Contact

- Email: [kasodariya.r@northeastern.edu](mailto:kasodariya.r@northeastern.edu)
- LinkedIn: [RohanKasodariya](https://www.linkedin.com/in/rohankasodariya/)
- GitHub: [RohanKasodariya](https://github.com/RohanKasodariya)

Feel free to reach out if you have any questions or suggestions!

---

Thanks for checking out this project! 😊

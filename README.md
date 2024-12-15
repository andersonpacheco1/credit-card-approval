# Credit Card Approval Application with Machine Learning

## Table of Contents

- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Objective](#objective)
- [Prerequisites](#prerequisites)
- [How to Use](#how-to-use)
- [Dataset](#dataset)
- [Model Methodology](#model-methodology)
- [License](#license)
- [Author and Contact](#author-and-contact)

## Project Description

This project involves the creation of a Streamlit application for credit approval analysis for individuals using a machine learning model. The goal is to simplify the credit approval process by allowing users to input their information and receive a predictive analysis regarding their credit approval based on a trained model.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, Scikit-learn
- **Deployment Framework**: Streamlit
- **Other Tools**: Jupyter, Git

## Objective

The objective of this application is to streamline the credit approval process by providing users with a predictive analysis of whether their credit application will be approved, based on a trained machine learning model.

## Prerequisites

You can install the required dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/andersonpacheco1/credit-card-approval
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to the `app` folder and run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

4. Alternatively, access the application online:
   [Credit Card Approval Streamlit App](https://credit-card--approval.streamlit.app/)

## Dataset

The dataset used for training the machine learning model is available on Kaggle:

- [Credit Card Approval Prediction Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

## Model Methodology

The machine learning model used for predicting credit card approval is a **Gradient Boosting Classifier**, which provided the best performance in terms of predictive accuracy. The model was trained using various features in the dataset, including applicant's demographic and financial information.

## License

This project is licensed under the MIT License.

## Author and Contact

- **Author**: Anderson Pacheco da Silva
- **Email**: anderson.we@outlook.com
- **LinkedIn**: [Anderson Pacheco da Silva](https://www.linkedin.com/in/andersonpachecodasilva/)

---
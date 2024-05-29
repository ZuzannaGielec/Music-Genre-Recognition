# Music Genre Classification Project

**Authors:** [Filip Chmiel](https://github.com/FChmiel242238?tab=overview&from=2024-05-01&to=2024-05-29), [Zuzanna Gielec](https://github.com/ZuzannaGielec)

## Abstract
This report presents a comprehensive study on the classification of music genres using a variety of machine learning models. We have exploited the GTZAN dataset from Kaggle, featuring audio files across ten distinct genres, to train and test our models, including Decision Trees, Random Forest, Gradient Boosting, and Convolutional Neural Networks. Our analysis focused on evaluating the efficacy of each model, fine-tuning hyperparameters, and comparing the outcomes against each other. The results offer insights into the potential of ensemble methods and deep learning approaches in processing and classifying complex audio data. This research contributes to the ongoing efforts in the field of Music Information Retrieval (MIR), demonstrating the capabilities and limitations of current methodologies.

## 1. Introduction

The ability to accurately classify music by genre is a significant challenge in the field of Music Information Retrieval, bearing important applications in music recommendation, archiving, and streaming services. This project addresses the task of Song Genre Classification from Audio Data, aiming to explore the effectiveness of various machine learning approaches in this domain. The motivation behind this study is twofold: to enhance the user experience by improving recommendation algorithms and to contribute to the academic discourse on audio classification methods. Previous work in this area has provided a solid foundation, yet there is room for enhancement, particularly in terms of accuracy and processing efficiency. This report describes the application of traditional machine learning models and a Convolutional Neural Network to the problem, discusses their performance, and suggests improvements over existing methods.

## 2. Materials and Methods

### 2.1. Data

We used a dataset called GTZAN taken from Kaggle, where it was uploaded by user Andrada. It consists of 1000 audio files in the .wav format, each 30 seconds in length. The tracks belong to 10 different genres: blues, classical, country, disco, hip hop, jazz, metal, pop, reggae, and rock. The Kaggle database additionally contains two Excel spreadsheets with various features pulled from the spectrograms of the audio files. Additionally, the database contains spectrograms for each audio file in the .png format.

### 2.2. Methods

In our experiments, we used three decision tree-based models using numerical features to classify our data: Decision Tree, Random Forest, and Gradient Boosting. Additionally, we trained a Convolutional Neural Network on spectrograms generated from the audio files.

### 2.3. Tools and Technologies

The entirety of our scripts has been written in Python 3.11.6 using various libraries, including pandas, scikit-learn, librosa, NumPy, Matplotlib, Keras, Graphviz, and SciPy.

### 2.4. Experimental Setup

We conducted experiments with various models and hyperparameters, focusing on accuracy as the main evaluation metric.

## 3. Results

We obtained results for each model and compared their performance using accuracy metrics. Here are the summarized results:

- **Decision Tree:**
  - Maximum Test Accuracy: 0.52
  - Average Test Accuracy: 0.43
  - Average Train Accuracy: 0.70

- **Random Forest:**
  - Maximum Test Accuracy: 0.71
  - Average Test Accuracy: 0.66
  - Average Train Accuracy: 0.99

- **Gradient Boosting:**
  - Maximum Test Accuracy: 0.69
  - Average Test Accuracy: 0.62
  - Average Train Accuracy: 1.00

- **Convolutional Neural Network:**
  - Test Accuracy: 0.6210
  - Train Accuracy: 0.7688

## 4. Discussion

The results obtained from the series of experiments highlight the distinctive strengths and weaknesses of each machine learning model employed. The Decision Tree, Random Forest, and Gradient Boosting methods showed varying degrees of success, with the ensemble methods outperforming the singular Decision Tree. However, the Convolutional Neural Network, typically robust in image classification tasks, demonstrated a modest accuracy, suggesting that audio data presents unique challenges that differ from visual data processing. This discussion underscores the complexity of audio feature extraction and the necessity of tailored models for audio classification tasks. 

## 5. Conclusions

In summary, this report has explored the application of various machine learning techniques to the task of song genre classification. It is evident that while traditional methods like Decision Trees provide a baseline understanding, ensemble methods such as Random Forest and Gradient Boosting offer substantial improvements. The application of a Convolutional Neural Network, though less effective than anticipated, opens avenues for future exploration into deep learning architectures specifically designed for audio data. Future work should focus on integrating domain-specific knowledge into the models and exploring the potential of unsupervised and semi-supervised learning paradigms.

## 6. Authors Contributions

- **Filip Chmiel:** Methodology, Software, Investigation, Writing – Original Draft
- **Zuzanna Gielec:** Validation, Visualization, Writing - Review & Editing

## 7. References

- scikit-learn documentation: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn documentation: https://scikit-learn.org/stable/modules/ensemble.html


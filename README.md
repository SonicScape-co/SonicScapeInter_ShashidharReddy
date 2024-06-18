# SonicScape-Intern-Work
<details>
  <summary>Signal Processing Techniques for Similarity Checks Between Patterns</summary>
  
# Signal Processing Techniques for Similarity Checks Between Patterns

## Cross Correlation
- Measure of similarity of two series as a function of displacement of one relative to the other.
- Ranges from -1 to +1.

## Dynamic Time Warping (DTW)
- Compares two temporal sequences that don't perfectly sync up through mathematics.
- Uses adaptive time normalization to create a warping path for sequences with different lengths and speeds.
- Requires matching every index from the first sequence with one or more indices from the other sequence.

## Fast Fourier Transform (FFT)
- Decomposes the original sequence of length N into a series of short sequences.
- Transforms signals from the time domain to the frequency domain.
- Frequency components can be used to identify similarity.

## Wavelet Transform
- Efficient method for evaluating small waves.
- Includes two transformation techniques: Continuous Wavelet Transform (CWT) and Discrete Wavelet Transform (DWT).
- Decomposes a signal into a set of basis functions (wavelets) that can analyze at various scales.

## Short Time Fourier Transform (STFT)
- Computes the Fourier transform of short, overlapping windows of the signal to analyze frequency over time.
- Used to determine sinusoidal frequency and phase content of local sections of a signal.
- Provides a smoother and more accurate frequency spectrum compared to FFT.

## Autocorrelation
- Measures the similarity of a signal with a delayed version of itself to identify repeating patterns.
- Not appropriate for comparing two different signals.

## Spectral Coherence
- Measures coherence between two signals in the frequency domain to identify common frequency components.
- Commonly used to estimate power transfer between input and output of a linear system.
- Tests for similar frequency components to determine the degree of linear dependency between signals.

## Singular Value Decomposition (SVD)
- Decomposes the data matrix into its constituent parts to identify common patterns.

## Principal Component Analysis (PCA)
- Reduces the dimensionality of data while preserving information to identify similarity.
- Similar datasets will have similar principal components.

## Dynamic Mode Decomposition (DMD)
- Data-driven analysis decomposes complex, non-linear systems into modes revealing underlying patterns and dynamics through spectral analysis.
- Used for dimensionality reduction, pattern recognition, noise reduction, and anomaly detection.

## Empirical Mode Decomposition (EMD)
- Decomposes signals into a set of oscillatory components called Intrinsic Mode Functions (IMFs) to analyze similarities.
- Useful for non-stationary signals.
- Output remains in the time spectrum and is not based on sine waves like FFT.

## Envelope Analysis
- Targets amplitude variation in vibration signals.
- Steps include shifting the frequency range in the high-frequency band to the base band, filtering the frequency-shifted signal using a low-pass filter, and calculating the envelope signal of the filtered signal.

## Hilbert Transform
- Computes instantaneous frequency and amplitude of a signal to analyze similarity in the time-frequency domain.
- Imparts a phase shift of +90 or -90 degrees to every frequency component of a function.
- Used to remove rapid oscillations from a signal to produce a direct representation of the envelope.

## Cosine Similarity
- Measures similarity between different data points or signals by calculating the cosine of the angle between two signals.
- Measures similarity based on the orientation of the signal.

## Symbolic Aggregate Approximation (SAX)
- Approximates time series data as a sequence of symbols to reduce dimensionality while preserving important characteristics.
- Reduces index dimension by using the boundary distance measure, which is lower than the Euclidean distance.

## Symbolic Bispectra based Lempel Ziv Complexity
- Combines symbolic representation with bispectral analysis, which examines the interaction between different frequency components of data.

</details>

<details>
  <summary>Machine Learning Algorithms and Techniques</summary>

  # Machine Learning Techniques

## Machine Learning Types

1. **Supervised Learning**
   - A) Continuous target variable → Regression → House price prediction
   - B) Categorical target variable → Classification → Medical imaging

2. **Unsupervised Learning**
   - A) Clustering → Customer segmentation
   - B) Association → Market basket analysis

3. **Semi-Supervised Learning**
   - A) Categorical target variable
     - Classification → Text classification
     - Clustering → Lane finding on GPS data

4. **Reinforcement Learning**
   - A) Categorical target variable
     - Classification → Optimized marketing
   - B) No target variable available
     - Control → Driverless cars

## Machine Learning Models

1. **Linear Regression**
   - Data analysis technique that predicts the value of unknown data by using another related and known data.
   - Example: Real estate company wants to predict the selling prices of houses based on various factors.

2. **Ridge Regression**
   - Statistical regularization technique (L2 regularization). Adds a penalty to the cost function to reduce overfitting.

3. **Lasso Regression**
   - Statistical regularization technique (L1 regularization). Adds the absolute value of the magnitude of the coefficient as a penalty term to the loss function.

4. **Elastic Net Regression**
   - Linear regression technique that uses a penalty term as both L1 and L2 norms weighted by a parameter called alpha. Useful for datasets with many predictors and multicollinearity.

5. **Logistic Regression**
   - Used to find answers to questions that have two or more finite outcomes. Appropriate when the total count has an upper limit and initial growth is exponential. Used for binary classification where the sigmoid function takes input as independent variables.
   - Example: Marketing research firm uses it to predict the likelihood of a customer purchasing a product based on their age, income, and education level.

6. **Decision Tree**
   - Non-parametric supervised algorithm used for both classification and regression tasks. Represents a series of decisions and their possible consequences.
   - Example: Decision tree is like a flow chart that helps a person decide what to wear based on weather conditions.

7. **Random Forest**
   - Ensemble technique that combines the output of multiple decision trees to reach a single result. Can handle both classification and regression problems.
   - Example: Classifies whether an email is spam or not. For regression problems, uses features like size, number of bedrooms, location, and age to predict the selling price of houses based on these features.

8. **Gradient Boosting Machine**
   - Combines the predictions from multiple decision trees by building a model in a stage-wise manner from weak learners and improves by correcting errors of previous models.
   - Example: Used in financial forecasting to predict stock prices by combining multiple decision trees, with each iteration refining predictions based on errors from earlier models. Both random forest and gradient boosting are ensemble techniques.

9. **XGBoost**
   - Scalable distributed gradient boost that provides parallel tree boosting and is one of the most used libraries.
   - Features include regularization, parallel processing, handling missing values, tree pruning, and built-in cross-validation. Can be used in high-stakes applications like fraud detection.

10. **LightGBM**
    - Faster training speed and higher efficiency. Uses a histogram-based algorithm that buckets continuous feature values into discrete bins, which speeds up the training procedure and uses low memory usage.

11. **CatBoost**
    - Designed for use on problems like regression with a very large number of independent features. Offers the highest predictive accuracy but is computationally expensive.
    - Works with categorical data and gradient boosting, hence the name CatBoost.

12. **AdaBoost**
    - Combines multiple weak learners to create a strong classifier. Each model is trained with a weighted dataset emphasizing instances that the previous model misclassified.

13. **Support Vector Machine (SVM)**
    - Supervised machine learning algorithm that classifies data by finding an optimal line or hyperplane that maximizes the margin between classes in an N-dimensional space.
    - Trained using several kernel functions:
      - A) Linear kernel function
      - B) Quadratic kernel function
      - C) Gaussian radial basis kernel function
      - D) Multilayer perceptron kernel function
    - Example: Used for handwriting recognition, intrusion detection, email classification, etc. Provides high-quality results but can be slow for non-linear problems and big data.

14. **K-Nearest Neighbor (KNN)**
    - Machine learning algorithm that uses proximity to compare one data point with a set of data it was trained on and has memorized to make predictions.
    - The most commonly used distance metric is Euclidean distance, where the nearest neighbor is assigned based on sorted distances.
    - Small K values lead to low bias and high variance.

15. **Principal Component Analysis (PCA)**
    - Dimensionality reduction method often used to reduce the dimensionality of large datasets by transforming a large set of variables into a smaller one that still contains most of the information.

16. **Independent Component Analysis (ICA)**
    - Technique used to separate mixed signals into their independent sources.

17. **Non-Negative Matrix Factorization (NMF)**
    - Matrix V is factorized into W (basis matrix) and H (coefficient matrix). This method constrains entries to be non-negative and helps in rank reduction.
    - Example: Used in image processing where NMF decomposes the image to assist in feature extraction and noise reduction.

18. **Gaussian Mixture Model (GMM)**
    - Soft clustering technique used in unsupervised learning to determine the probability that a given data point belongs to a cluster. Estimates mean and covariance of the components.
    - Applications include anomaly detection, clustering, and density estimation.

19. **Hierarchical Clustering**
    - Algorithm that groups similar objects into clusters hierarchically. Ends with a set of clusters where each cluster is distinct from others, and objects within each cluster are broadly similar.
    - Example: Clustering four cars into two clusters based on car types like sedan and SUV.

20. **Mean Shift Clustering**
    - Shifts each data point towards the highest density of distribution points within a certain radius.

21. **Agglomerative Clustering**
    - Begins with N groups, each containing one entity, and then merges the two most similar groups at each stage until a single group containing all data is reached.

22. **Feedforward Neural Networks**
    - Information flows in only one direction from input nodes through hidden nodes to output nodes.
    - Consists of an input layer, hidden layers, and an output layer.
    - Example: Used for image classification where an image is input, and the model predicts the class label of the image.

23. **Convolutional Neural Networks (CNN)**
    - Includes convolutional layers, max-pooling layers, and fully connected layers. Used to detect and classify objects in an image.

24. **Recurrent Neural Networks (RNN)**
    - Output from the previous step is fed as input to the current step.
    - Example: Can create a language translator with an RNN that analyzes a sentence and correctly structures the words in a different language.

25. **Long Short-Term Memory (LSTM)**
    - Strong ability to learn and predict sequential data. Has input, forget, and output gates for processing sequences.
    - Applications include speech recognition, time series prediction, etc.

26. **Gated Recurrent Unit (GRU)**
    - Similar to LSTM but with a simpler architecture and fewer parameters. Designed to model sequential data by selectively remembering or forgetting information over time.

27. **Autoencoders**
    - Special type of neural network trained to copy input to output. Used for tasks like noise reduction in images to improve quality.

28. **Variational Autoencoder (VAE)**
    - Enhanced form of an autoencoder that uses regularization techniques to overcome overfitting and ensure desirable properties.

29. **Generative Adversarial Network (GAN)**
    - Learns to generate new data with the same statistics as a training set. Generates completely new images rather than just improving existing ones like autoencoders.

30. **Multi-Layer Perceptron (MLP)**
    - Neural network with multiple layers of nodes fully connected to each other. Used in natural language processing, image recognition, and speech recognition.

31. **Seq2Seq Model**
    - Attention model that allows the decoder to focus on the most relevant parts of the input sequence. Boosts accuracy in sequence-to-sequence tasks.

32. **Word Embedding Model**
    - Representation of words as vectors in a multi-dimensional space where the distance and direction between vectors reflect the similarity and relationship among corresponding words.

33. **ARIMA (AutoRegressive Integrated Moving Average)**
    - Time series analysis technique for forecasting future values based on past values. Uses lagged moving averages to smooth time series data.
    - Used in technical analysis to forecast future security prices.

34. **Seasonal ARIMA**
    - Extension of ARIMA that includes seasonality in addition to non-seasonal components.

35. **Dynamic Time Warping (DTW)**
    - Measure of similarity between two temporal sequences that may vary in speed. Replaces Euclidean distance by allowing many-to-one comparisons.

36. **Hidden Markov Model (HMM)**
    - Predicts future values based on current and previous values. Captures underlying patterns in sequential data.

37. **Naive Bayes**
    - Probabilistic classifier based on applying Bayes' theorem with naive assumptions between features. Assumes the presence of a particular feature in a class is independent of the presence of any other feature.


</details>
<details>
<summary>Varying factors affecting data</summary>

<br> 
<details>
<summary>General factors that influence data quality</summary>

1) **Research Design** - Methods of data collection and qualitative & quantitative approach.

2) **Sampling Strategy** - Methods used like random, stratified, convenience sampling impact data variability.

3) **Data Collection Methods** - Techniques employed influence data type and quality. Also, the environment affects the quality of data.

4) **Measurement Tools** - Instruments ensure validity and reliability of data.

5) **Timing** - Collection timing influences data relevance.

6) **Researcher Bias** - Personal perspectives can unintentionally affect data collection and analysis.

7) **Resources** - Availability of time, budget, and technology affect scope and feasibility.

8) **Data Quality Control** - Procedures ensure accuracy, completeness, and consistency of collected data.
</details>


<details>
  <summary>General factors that influence data quality collected from vibration sensors</summary>

1) **Sensor Selection** - The type and quality of sensors chosen for vibration measurement can significantly impact the accuracy and range of data collected. Factors such as frequency response, sensitivity, resolution, and durability of the sensors are crucial considerations.

2) **Placement of Sensors** - Proper placement of sensors on the equipment or structure being monitored is critical. Sensors should be positioned to capture vibrations accurately and representatively, considering factors like mounting surface, orientation, and proximity to vibration sources.

3) **Sampling Rate** - The rate at which data is sampled (sampling frequency) affects the level of detail captured in the vibration signals. Higher sampling rates capture more nuances in vibrations but may require more storage and processing resources.

4) **Signal Conditioning** - Pre-processing of the vibration signals through filtering, amplification, and noise reduction techniques can improve the quality of the data collected, enhancing the accuracy of subsequent analysis.

5) **Calibration** - Regular calibration of sensors ensures that they are measuring vibrations accurately over time. Calibration helps maintain data integrity and reliability, especially in long-term monitoring applications.

6) **Environmental Conditions** - Factors such as temperature, humidity, and electromagnetic interference (EMI) can influence sensor performance and data quality. Monitoring and controlling these environmental variables are essential for reliable data collection.

7) **Data Synchronization** - When using multiple sensors or integrating vibration data with other types of data (e.g., temperature, pressure), synchronization ensures that all data points are aligned in time, avoiding discrepancies in analysis.


</details>

<details>
  <summary>Factors that influence pattern of relationship in data</summary>

1) **Equipment Variation** - Differences calibration of the sensors used to measure temperature and acceleration can introduce variability in the data.

2) **Operational Conditions** - Variations in operational parameters such as speed, load, or environmental conditions (humidity, atmospheric pressure) can influence both temperature and acceleration readings.

3) **Sampling Frequency** - The rate at which data is sampled affects the resolution and detail of patterns captured in the datasets.

4) **Seasonal Variations** - Seasonal changes can impact temperature readings and may influence patterns observed in acceleration data, especially in outdoor or environmental monitoring.

5) **Vibration Sources** - Specific machinery or processes generating vibrations can cause distinct patterns in acceleration data across different axes.

6) **Temperature Effects on Material Properties** - Temperature changes can affect material properties, potentially influencing vibration patterns and acceleration readings.

7) **Structural Dynamics** - The natural frequencies and modal characteristics of structures being monitored can affect acceleration patterns, especially in structural health monitoring applications.

8) **Data Synchronization** - Ensuring that time stamps align correctly across temperature and acceleration datasets is crucial for accurate analysis of temporal correlations and patterns.

</details>
</details>

Project Title: ADAS for Two-Wheeler Drivers

Description: This project, undertaken as part of the Bachelor of Technology degree at the Institute of Infrastructure Technology Research and Management, focuses on developing an Advanced Driver Assistance System (ADAS) tailored for two-wheeler drivers. By integrating smart wearables and artificial intelligence, the system aims to enhance safety and assistance for riders through real-time monitoring of driver behavior.

Contributors: Mihir Vangani, Kunj Patel

Objective: The project aims to collect sensor data from smartwatches and wearables, including parameters like acceleration and heart rate, to train an AI model capable of detecting instances where the driver may require attention or assistance. This involves developing custom firmware, application development for smartwatches, and utilizing the Google Fit API for data collection.

Supervisor: Prof. (Dr.) Manish Chaturvedi

Methedology: Our project endeavors to revolutionize road safety by leveraging vehicular and physiological sensors to assess a two-wheeler driver's behavior comprehensively. We employ a sophisticated system where sensors, including the MAX30100 heart rate sensor, MPU6050 accelerometer and gyroscope, and Neo-6M GPS module, are powered by a power bank. During a 20-minute drive, data is continuously collected at a frequency of 32 Hz.

Four switches, each representing distinct driving scenarios such as normal driving, rash driving, normal turn, and rash turn, enable precise data categorization. Switch activation logic is applied to the corresponding sensor data, facilitating differentiation between various driving behaviors. If no switch is activated, data is recorded as NAN.

To ensure uniformity, data is down-sampled to a constant 30 Hz frequency, and noise is mitigated through Low Pass Filter (LPF) and Kalman Filter techniques. Following data merging and noise removal, feature extraction ensues, yielding a Feature Matrix. Principal Component Analysis (PCA) is then employed to reduce dimensionality, allowing for visualization using T-distributed Stochastic Neighbor Embedding (t-SNE).

The extracted features encompass statistical measures such as mean, standard deviation, median absolute deviation, energy measure, interquartile range, signal entropy, skewness, and kurtosis. These features encapsulate the nuances of driving behavior captured by the sensors.

Subsequently, a variety of classifiers, including Logistic Regression, Linear Discriminant Analysis, KNN, Gaussian Naive Bayes, Linear Support Vector Classification, Decision Tree, Random Forest, and XGBoost, are applied to the dataset for comprehensive analysis and prediction.

Through meticulous data collection, feature extraction, dimensionality reduction, and classifier application, our project aims to contribute significantly to the development of advanced driver assistance systems (ADAS) and road safety technologies. This holistic approach holds the promise of enhancing two-wheeler driver safety and mitigating the risks associated with hazardous driving behaviors.

Completion Date: October 2023

Certification:The project report has been submitted in partial fulfillment of the requirements for the Bachelor of Technology degree in the Department of Electrical Engineering at the Institute of Infrastructure Technology Research and Management, Ahmedabad.

Note: This project's objective is to contribute to the advancement of safety systems for two-wheeler drivers and lay the foundation for future development of ADAS technologies tailored for the specific needs of this user group.

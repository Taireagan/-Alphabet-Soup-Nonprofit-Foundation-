<div align="center">
    <h1>Alphabet Soup</h1>
</div>

<div align="center">
    <h3>Nonprofit Foundation</h3>
</div>

<div align="center">
    <h4>By: Tai Reagan</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/charity_funding.jpg" alt="charity_funding" width="700"/>
</div>

---
<a name="top"></a>
# Table of Contents

<details>
  <summary>Click To Expand</summary>

- [Alphabet Soup Background](#alphabet-soup-background)
- [Resources](#resources)
- [Stages of the Machine Learning and Neural Network Process](#stages-of-the-machine-learning-and-neural-network-process)
  - [Preprocessing The Data](#preprocessing-the-data)
      - [Removing Columns](#removing-columns)
      - [Identifying Unique Values and Cutoff Point](#identifying-unique-values-and-cutoff-point)
      - [Create Training and Testing Datasets](#create-training-and-testing-datasets)
      - [Scaling The Data](#scaling-the-data)
  - [Compile, Train, and Evaluate the Model](#compile-train-and-evaluate-the-model)
  - [Model Optimization](#model-optimization)
    - [Optimization 1](#optimization-1)
    - [Optimization 2](#optimization-2)
    - [Optimization 3](#optimization-3)
- [Summary](#summary)
- [Recommendation](#recommendation)
</details>



## Alphabet Soup Background
Alphabet Soup, a nonprofit foundation, seeks a data-driven tool to improve its funding decisions by identifying applicants with the highest potential for successful outcomes in their ventures. To support this initiative, Alphabet Soup’s business team provided a CSV dataset with information on over 34,000 organizations that have received funding from the foundation over the years. This dataset contains multiple columns capturing key metadata for each organization, such as organizational attributes, historical performance indicators, and other relevant features. 

Leveraging machine learning and neural networks, this initiative develops a binary classifier that evaluates these features to predict an applicant’s likelihood of success if funded. By building this predictive model, Alphabet Soup aims to allocate its resources more effectively, prioritizing applicants with promising success potential and enhancing the overall impact of its funding initiatives.


## Resources 
The data used for the analysis was provided by [Charity Data](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv)


---

## Stages of the Machine Learning and Neural Network Process
### Preprocessing The Data
To ensure accurate model performance, it is essential to preprocess the dataset using tools like Pandas and scikit-learn’s StandardScaler(). Data preprocessing involves preparing raw data by cleaning and transforming it into a format that machine learning models can interpret effectively. This step is crucial, as unprocessed data can introduce inconsistencies, biases, or scaling issues that may lead to inaccurate predictions. The process begins by uploading the provided starter file to Google Colab and reading in the charity data into a Pandas DataFrame.

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/read_in_csv.png" alt="read_in_csv" width="600"/>
</div>

<details>
  <summary>About The Data</summary>

  - **EIN** and **NAME**: Identification columns
  - **APPLICATION_TYPE**: Alphabet Soup application type
  - **AFFILIATION**: Affiliated sector of industry
  - **CLASSIFICATION**:Government organization classification
  - **USE_CASE**: Use case for funding
  - **ORGANIZATION**: Organization type
  - **STATUS**: Active status
  - **INCOME_AMT**: Income classification
  - **SPECIAL_CONSIDERATIONS**: Special considerations for application
  - **ASK_AMT**: Funding amount requested
  - **IS_SUCCESSFUL**: Was the money used effectively
</details>

This step provides a clear understanding of the types of columns within the dataset, enabling the identification of target variables and feature variables essential for the model.

<details>
  <summary>Target Variable</summary>

  - **IS_SUCCESSFUL**: Was the money used effectively
</details>

<details>
  <summary>Feature Variables</summary>

  - **APPLICATION_TYPE**: Alphabet Soup application type
  - **AFFILIATION**: Affiliated sector of industry
  - **CLASSIFICATION**:Government organization classification
  - **USE_CASE**: Use case for funding
  - **ORGANIZATION**: Organization type
  - **STATUS**: Active status
  - **INCOME_AMT**: Income classification
  - **SPECIAL_CONSIDERATIONS**: Special considerations for application
  - **ASK_AMT**: Funding amount requested
</details>

### Removing Columns
After identifying the target and feature variables remove columns that will not contribute to the model. In this case, the EIN and NAME columns are excluded from the analysis.

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/drop_columns.png" alt="drop_columns" width="600"/>
</div>

### Identifying Unique Values and Cutoff Point
The next step involves identifying the number of unique values in each column and calculating the frequency of each unique value for columns with more than 10 distinct entries. In this dataset, the APPLICATION_TYPE and CLASSIFICATION columns meet this criterion. For these columns, a cutoff point will be selected to consolidate "rare" categories into a new category labeled Other. After this transformation, we will verify that the replacement was implemented successfully.

<div align="center">
    <h4>Application Type</h4>
</div>


<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/types_cuttoff.png" alt="application_type_cutoff" width="600"/>
</div>

<div align="center">
    <h4>Classification</h4>
</div>


<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/class_cutoff.png" alt="classification_cutoff" width="600"/>
</div>


<details>
  <summary>More Info</summary>

When dealing with categorical data, it is important to handle "rare" categories to ensure that the model 
can generalize well and avoid being influenced disproportionately by infrequent values. In the step above, 
using the number of data points for each unique value helps determine a cutoff point for these rare categories.
Categories with counts below this threshold are grouped together under a new label, such as "Other."

This consolidation improves model performance in several ways:

**Reduces Noise:** Rare categories often introduce noise, which can lead to overfitting. Grouping them together helps the model focus on the patterns in more frequently occurring categories.

**Improves Efficiency:** Combining rare categories reduces the number of unique values, making the model training process more efficient and reducing the complexity of the dataset.

**Ensures Sufficient Data for Each Category:** Machine learning algorithms perform better with larger sample sizes for each category. By grouping rare categories, each category used in training has enough data to contribute meaningfully to the model.
</details>

### Create Training and Testing Datasets

Before creating the training and testing datasets, **pd.get_dummies()** is used to encode categorical variables. This function converts categorical data into a numerical format suitable for machine learning algorithms, which typically require numeric inputs. This process, called one-hot encoding, generates a new binary column for each unique category within the original categorical column, enabling the model to interpret categorical data effectively.

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/pd_dummies.png" alt="pd_dummies" width="600"/>
</div>

After encoding the categorical variables, the preprocessed data is divided into a features array, X, and a target array, y. These arrays are then used with the **train_test_split** function to partition the data into training and testing datasets.

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/split_data.png" alt="split_data" width="600"/>
</div>

### Scaling The Data

The final step in preparing the data before compiling, training, and evaluating the neural network model is to scale the features in both the training and testing datasets. This is achieved by creating a **StandardScaler** instance, fitting it to the training data, and then applying the **transform** function. Scaling ensures that all features are on a similar scale, which enhances model stability and performance during training.

By partitioning the data into training and testing sets, the model can be trained on one portion (the training set) and evaluated on an unseen portion (the testing set). This approach is a best practice in machine learning, providing an unbiased estimate of model performance and contributing to reliable, generalizable predictions.

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/standard_scaler.png" alt="standard_scaler" width="600"/>
</div>

 Once the preprocessing steps are then completed to prepare the data the next phase is compiling, training, and evaluating the neural network model.


[Back to Top](#top)


### Compile, Train, and Evaluate the Model
Using TensorFlow, a neural network, or deep learning model, is designed to build a binary classification model that predicts the likelihood of success for organizations funded by Alphabet Soup, based on features in the dataset. The process begins by analyzing input features to determine the optimal number of neurons and layers for the model architecture. After defining the model structure, it is compiled, trained, and evaluated to assess its performance using metrics such as loss and accuracy, providing valuable insights into the model’s predictive effectiveness.

<details>
  <summary>Steps For Compiling, Training, and Evaluating the Model </summary>

  1. Continue using the file in Google Colab in which preprocessing steps were performed from inital start.
  2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
  3. Create the first hidden layer and choose an appropriate activation function.
  4. If necessary, add a second hidden layer with an appropriate activation function.
  5. Create an output layer with an appropriate activation function.
  6. Check the structure of the model.
  7. Compile and train the model
  8. Create a call back that saves the model's weights every five epochs
  9. Evaluate the model using the test data to determine the loss and accuracy.
  10. Save and export your results to an HDF5 file. 
</details>

> [!TIP]
> Epoch is one complete pass through the entire dataset during model training, allowing the model to learn and adjust its parameters with each cycle to improve accuracy.


## Model Optimization
The three models were developed with the goal of achieving a target predictive accuracy higher than 75%, each progressively optimized using one or more of the following techniques:

<details>
  <summary>Optimization Methods</summary>

  - Dropping more or fewer columns
  - Creating more bins for rare occurrences in columns
  - Increasing or decreasing the number of values for each bin
  - Adding more neurons to a hidden layer
  - Adding more hidden layers
  - Using different activation functions for the hidden layers
  - Adding or reducing the number of epochs to the training regimen
</details>

### Optimization 1
For the first model it consists of an input layer (matching the number of input features), two hidden layers (with 80 and 30 neurons, respectively), and an output layer with a single neuron using a sigmoid activation function for binary classification. The relu activation function is applied to the hidden layers to introduce non-linearity. You can find definitions in the drop down menu below.

<details>
  <summary>Definitions</summary>

  - **Input Layer**: Is the first layer of a neural network, responsible for receiving the input data and passing it to the next layer for further processing. Its size corresponds to the number of features in the dataset.
  - **Hidden Layers**: Is an intermediate layer between the input and output layers, where the model learns complex patterns by processing and transforming data through weighted connections and activation functions
  - **Neurons**: Are the fundamental units in a neural network layer that process and pass information through the network. Each neuron performs a calculation on the data it receives, applying a weight, bias, and activation function to detect patterns.
  - **Sigmoid Activation Function**: Is a mathematical function that transforms inputs into an output between 0 and 1, making it ideal for binary classification tasks where predictions represent probabilities.
  - **Relu Activation Function**: Outputs the input directly if it’s positive and zero otherwise, helping neural networks learn complex patterns by introducing non-linearity while being computationally efficient.
</details>


<div align="center">
    <h4>Defining The Model</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/define_model.png" alt="define_moodel" width="600"/>
</div>

<div align="center">
    <h4>Model Output</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/summary_output.png" alt="summary_output" width="600"/>
</div>

This model architecture consists of three fully connected (dense) layers: the first with 80 neurons, the second with 30 neurons, and a final output layer with 1 neuron for binary classification. The model has a total of 5,981 trainable parameters, which are adjusted during training to optimize prediction accuracy. Each layer's output shape reflects the number of neurons, with flexibility in batch size for efficient processing.

<div align="center">
    <h4>Training The Model</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/train_model.png" alt="train_model" width="600"/>
</div>

This step trains the model over 100 epochs, using 85% of the data for training and 15% for validation. The output shows the model's accuracy and loss improving with each epoch, helping to monitor its performance on both training and validation sets.

<div align="center">
    <h4>Evaluate The Model</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/evaluate.png" alt="evaluate" width="600"/>
</div>


This step evaluates the trained model on the test dataset to measure its performance on unseen data. The output shows a test accuracy of approximately 72.8% and a loss of 0.5666, indicating how well the model generalizes and the error rate on new data. This final evaluation helps validate the model's effectiveness in making predictions.


[Back to Top](#top)



### Optimization 2


<div align="center">
    <h4>Defining The Model</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/define_model_opt_1.png" alt="define_moodel_opt_1" width="600"/>
</div>


To improve upon the initial test accuracy of 72.8%, the model was adjusted in the second optimization attempt by reducing the number of neurons in each layer and adding an additional hidden layer. Specifically, the first and second hidden layers were reduced to 7 and 14 neurons, respectively, which decreases the total number of parameters (weights), making the model more efficient. However, this reduction in neurons could limit the model's capacity to capture complex patterns within a single layer.

To address this, a third hidden layer with 21 neurons was added, enhancing the model’s ability to learn hierarchical patterns and capture complex relationships in the data. This additional layer introduces depth, allowing the model to combine simpler features from earlier layers into more intricate patterns in later layers, which can potentially improve its predictive performance and generalization capability.

<div align="center">
    <h4>Model Output</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/summary_output_opt_1.png" alt="summary_output_opt_1" width="600"/>
</div>

The second model is simpler, with three hidden layers containing 7, 14, and 1 neurons, respectively, totaling only 435 trainable parameters. This streamlined model reduces computational load and is less likely to overfit but may limit the model’s ability to capture complex patterns compared to the first model. The second model’s structure is more efficient, prioritizing generalization and faster training at the potential cost of accuracy.

<div align="center">
    <h4>Training The Model</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/train_model_opt_1.png" alt="train_model_opt_1" width="600"/>
</div>

Similar to the previous model training optimization 2 trains the model over 100 epochs, using 85% of the data for training and 15% for validation. The output shows the model's accuracy and loss improving with each epoch, helping to monitor its performance on both training and validation sets.

<div align="center">
    <h4>Evaluate The Model</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/Evaluate_opt_1.png" alt="Evaluate_opt_1" width="600"/>
</div>


In the second optimization attempt, the model achieved an accuracy of approximately 73.2% and a loss of 0.5536 on the test data, showing a slight improvement over the initial model's accuracy of 72.8% and loss of 0.5666. This minor increase in accuracy and decrease in loss suggests that the changes in model architecture — reducing neurons and adding a hidden layer — contributed to a marginal improvement in performance. However, the improvement is small, indicating that further optimization may be needed to achieve more substantial gains.


[Back to Top](#top)




### Optimization 3
For the third optimization trial, a different approach was taken by retaining more columns in the dataset during preprocessing. Dropping fewer columns helps preserve valuable information, allowing the model to access a broader range of features that may improve pattern detection and predictive accuracy. By retaining more columns, the model has a greater ability to capture complex relationships within the data, reducing the risk of overly simplistic results and enhancing its overall optimization potential.

<div align="center">
    <h4>Droping The EIN Column</h4>
</div>

In this section, rather than dropping both the **EIN** and **NAME** columns, only the **EIN** column was removed, while the **NAME** column was retained

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/drop_columns_opt_2.png" alt="drop_columns_opt_2" width="600"/>
</div>

<div align="center">
    <h4>View NAME Counts</h4>
</div>

As with the previous step for **APPLICATION_TYPE** and **CLASSIFICATION**, this step involves examining the counts in the **NAME** column to determine appropriate binning.

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/name_counts_opt_2.png" alt="name_counts_opt_2" width="600"/>
</div>

<div align="center">
    <h4>Identifying Unique Values and Cutoff Point</h4>
</div>

Here, a cutoff value was set for the NAME column, grouping names with fewer than 5 occurrences and replacing them with the label "Other." This reduces the number of unique categories, simplifying the data and enhancing model performance by directing focus toward more common entities.

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/names_count_cutoff_opt_2.png" alt="names_count_cutoff_opt_2" width="600"/>
</div>


<div align="center">
    <h4>Defining The Model</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/define_model_opt_2.png" alt="define_model_opt_2" width="600"/>
</div>


In the third optimization trial, the model remains the same as in the second attempt, but with an increased number of input columns.


<div align="center">
    <h4>Model Output</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/summary_output_opt_2.png" alt="summary_output_opt_2" width="600"/>
</div>

As a result, the summary output for the third optimization trial is identical to that of the previous optimization attempt.

<div align="center">
    <h4>Training The Model</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/train_model_opt_2.png" alt="train_model_opt_2" width="600"/>
</div>

Similar to the previous 2 model training optimizations trains the model over 100 epochs, using 85% of the data for training and 15% for validation. The output shows the model's accuracy and loss improving with each epoch, helping to monitor its performance on both training and validation sets.

<div align="center">
    <h4>Evaluate The Model</h4>
</div>

<div align="center">
    <img src="https://github.com/Taireagan/-Alphabet-Soup-Nonprofit-Foundation-/blob/main/Images/Evaluate_opt_2.png" alt="Evaluate_opt_2" width="600"/>
</div>

The third optimization attempt yielded evaluation results on the test data, with an accuracy of approximately 78.8% and a loss of 0.4691. These metrics demonstrate an improvement over previous trials, indicating that the adjustments made in the third optimization positively enhanced the model's predictive accuracy.



[Back to Top](#top)



### Summary

Across the three optimization attempts, the model progressively improved in accuracy and reduced loss. The initial model achieved an accuracy of 72.8% with a loss of 0.5666. In the second attempt, adjustments included reducing the number of neurons and adding a third hidden layer, yielding a slight improvement with an accuracy of 73.2% and a loss of 0.5536. These changes aimed to prevent overfitting while allowing the model to capture more complex patterns through an additional layer.

In the third optimization, the model remained the same as the second attempt, but additional input columns were retained, allowing the model access to a broader set of features. This led to a significant performance boost, with the model reaching an accuracy of 78.8% and a reduced loss of 0.4691. The retention of more columns provided the model with richer information, enhancing its ability to learn and detect patterns more effectively.

Overall, the incremental changes—adding layers, adjusting neuron counts, and preserving more input features—contributed to a marked improvement in the model’s accuracy and ability to generalize to new data.


[Back to Top](#top)



### Recommendation

To address this classification problem, a recommendation would be to consider using a Random Forest classifier as an alternative model. It could be a better option for several reasons:

1. **Robustness and Stability:** This approach reduces the risk of overfitting, which is a common issue with individual models like neural networks.
2. **Reduced Sensitivity to Hyperparameters:** Random Forests are less sensitive to hyperparameter tuning and generally perform well with minimal adjustments, making them easier to implement and optimize.
3. **Performance on Small to Medium Datasets:** Neural networks generally excel with large datasets, as they need substantial data to learn complex patterns effectively. Random Forests, however, perform well even with smaller datasets, making them suitable for problems where data may be limited or where collecting additional data is challenging.
4. **Handling of Mixed Data Types:** Random Forests can naturally handle datasets with both categorical and numerical features without requiring extensive preprocessing, such as scaling or encoding. Neural networks, in contrast, typically require scaling and other preprocessing steps to work effectively, especially with numerical inputs.

 Random Forest is an ensemble learning method that builds multiple decision trees and combines their outputs for more robust and stable predictions. This model can handle complex interactions between features and is less sensitive to overfitting than a single neural network layer. Additionally, Random Forests can be more interpretable than deep neural networks, allowing insight into feature importance, which could be useful in understanding which factors contribute most to the success of organizations funded by Alphabet Soup.



[Back to Top](#top)






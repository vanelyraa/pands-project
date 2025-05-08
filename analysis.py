# Programming and scripting final project
# Autor: Vanessa Lyra



# Imports

#Importing machine learning Scikit-Learn library and loading dataset
from sklearn import datasets

#Panda library
import pandas as pd

#Python package provides multi-dimensional array objects
import numpy as np

#Graph plotting library in Python
import matplotlib.pyplot as plt

#Importing seaborn library
import seaborn as sb

#Regression linear module from scikit-learn
from sklearn.linear_model import LinearRegression



## Loading the dataset

#Loading dataset to my repository
iris_ds = datasets.load_iris()



## 1. Exploring the dataset structure

# Returning dataset values
print(iris_ds) #Printing dataset content

# Using Pands DataFrame to convert Iris dataset into a "table" to facilitate data manipulation, dataset features set as columns
my_dataframe = pd.DataFrame(iris_ds.data, columns=iris_ds.feature_names)

#Dataset shape, printing the shape of the dataset
print(f'Iris dataset shape: {my_dataframe.shape}') 

# Dataset features, printing feature names from dataset
print(f' Feature names: {iris_ds['feature_names']}')

# Dataset target classes, printing dataset target classes and target names
print(f'a. Dataset target (classes): {iris_ds['target']}\n\n')
print(f'b. Dataset target names: {iris_ds['target_names']}')

#Identifying duplicated data in dataset
duplicate = my_dataframe.duplicated() #Storing results in variable duplicate
print(duplicate) #Printing results



## 2. Variables Summary

# Generating statistical information summary (max, min,mean,standart deviation and Q1, median, Q3 quartiles)
variable_summary = my_dataframe.describe()

# Save the summary to a text file
with open('variable_summary.txt', 'w') as my_file: #Creating file 'variable summary', if the file exists, Python opens it in write mode
    my_file.write('Summary of each Iris variable:\n\n') #Adding a file to 'txt file
    my_file.write(variable_summary.to_string())  # Convert DataFrame to string for readable format



## 3. Histograms

# Plotting histograms for each feature
for iris_features in my_dataframe.columns: #Looping dataset columns
    plt.hist(my_dataframe[iris_features], edgecolor='black') #Creating histograms of each feature
    plt.title(f'Histogram of {iris_features}') #Adding title

    #Adding X and Y axis labels
    plt.xlabel(f'{iris_features} histogram') #X label
    plt.ylabel(f'{iris_features} frequency')  #Y label
    plt.grid(axis = 'y', linestyle = '--', linewidth = 0.7) #Adding horizontal grid lines
    plt.savefig(f'{iris_features}_histogram.png') #Saving plots to .png
    plt.show()



    ## 4. Scatter plots

    # Loading Iris dataset using seaborn
iris_sb = sb.load_dataset('iris')

# Sepal Length vs Sepal Width
# data = source of data Iris dataset
# x/y = Chosen dataset columns for X and Y axix
# hue = python defines the color of the 'dots' based on species
sb.scatterplot(data=iris_sb, x='sepal_length', y='sepal_width', hue='species') #Scatterplot 
plt.title('Sepal Length x Sepal Width plot')
plt.show()

# Petal Length vs Petal Width
sb.scatterplot(data=iris_sb, x='petal_length', y='petal_width', hue='species')
plt.title('Petal Length x Petal Width plot')
plt.show()

# Sepal Length vs Petal Length
sb.scatterplot(data=iris_sb, x='sepal_length', y='petal_length', hue='species')
plt.title('Sepal Length x Petal Length plot')
plt.show()

# Sepal Length vs Petal Width
sb.scatterplot(data=iris_sb, x='sepal_length', y='petal_width', hue='species')
plt.title('Sepal Length x Petal Width plot')
plt.show()

# Sepal Width vs Petal Length
sb.scatterplot(data=iris_sb, x='sepal_width', y='petal_length', hue='species')
plt.title('Sepal Width x Petal Length plot')
plt.show()

# Sepal Width vs Petal Width
sb.scatterplot(data=iris_sb, x='sepal_width', y='petal_width', hue='species')
plt.title('Sepal Width x Petal Width plot')
plt.show()



## 5. Boxplots

# Looping dataset columns, except last column (species column)
for iris_features in iris_sb.columns[:-1]: 
    # Plotting box plot with seaborn library
    # Data: iris dataset from seaborn
    # X axis: flower calling column species from Seaborn dataFrame
    # Y axis: Dataset flower features 
    sb.boxplot(data=iris_sb, x='species', y=iris_features)

    # Using F-string to add title and label according to the feature being plotted
    plt.title(f'Boxplot of {iris_features}')
    plt.xlabel('Species')
    plt.ylabel({iris_features})
    
    # Plotting
    plt.show()



## 6. Heatmap

# Calculating correlation matrix from dataframe (table)
my_correlation = iris_sb.corr(numeric_only=True)

# Plotting heatmap
# annot = True: adding numerical values from the matrix into the heatmap cells. 
# cmap: defining a color palette to heatmap
sb.heatmap(my_correlation, annot=True, cmap='coolwarm')
plt.title('Iris Correlation Heatmap') # Adding title to heatmap
plt.show() # Displaying heatmap



## 7. Pairplots

# Creating pairplot
# hue: Species is the variable used to map plot features into different colors.
sb.pairplot(iris_sb, hue="species")

# Show the plot
plt.show()



## 8. Simple linear regression

# Adding a new column to dataset with flower species called 'iris_classes'
my_dataframe['iris_classes'] = iris_ds.target_names[iris_ds.target]

# Store column data (except the newly created 'iris_classes') in variable iris_features
iris_features = my_dataframe.columns[:-1]

# The Nested loops will iterate throught each dataset to define each pair to be compared from the dataset, stored in X and Y

#first loop iterates through the as many times as the length of iris_features
#Iris features length is 4 (four feature columns)
#The Loop will iterate 4 times through indexes 0,1,2,3 from iris_features
for i in range(len(iris_features)): 
    #This second loop with iterate 4 times as well, but first, it will sum 1 to to index given by i.
    for j in range(i + 1, len(iris_features)):
        #Accessing the column iterated in i/j
        #. values: converts the data into a array
        # .reshape: reshapes the data into a 2D array required for the regression
        x = my_dataframe[iris_features[i]].values.reshape(-1, 1)
        y = my_dataframe[iris_features[j]].values

        # Calling linear regression function
        model = LinearRegression().fit(x, y)
        #Generating the predicted values for X
        y_pred = model.predict(x)

        # Plot
        #Generating scatterplot using Seabornusing
        #data: where the data comes from, in this case from the Dataframe (dataset stored as 'table')
        #X: the X axis will be featured from i in variable iris_features (defined by first for loop)
        #Y: the Y axis will be featured from j in variable iris_features (defined by second for loop) 
        #Hue: will define diferent colors to the plor based on species stored in 'iris classes'
        sb.scatterplot(data=my_dataframe, x=iris_features[i], y=iris_features[j], hue='iris_classes')

        #Plotting the regression line
        #.plot: plots a line based on data determined by the user
        # defining the regression line color 'black' and adding label to plot
        plt.plot(my_dataframe[iris_features[i]], y_pred, color='black', label='Global Regression')
        plt.show()
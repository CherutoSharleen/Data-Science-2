# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read from the csv file
exams = pd.read_csv('exam_grades.csv')
exams


#columns and rows in the dataset
#information about the dataframe
exams.info()
#rows and column
print(exams.shape)
print(exams.shape[0])
len(exams.columns)
print(exams.shape[1])


# exams.head() #print first 5 values
#unique values on the student id column
print(exams['student_id'].unique())
print(exams.student_id.unique())
print(pd.unique(exams['student_id']))


#count Null Values in the dataframe under column grade
print(exams['grade'].isna().sum())
#Count all null values in the data frame
print("The missing values in the dataframe are ",exams.isna().sum().sum())
#Count all null values in row 1

#replacing empty grades with 0
exams['grade']=exams.grade.fillna(0)


#count grades higher than 70
results = exams[exams['grade'] >= 70]
results.count()

#find the mean per exam
exams.groupby(by='exam').grade.mean()

#distribution of exam grades histogram
plt.hist(exams.grade)



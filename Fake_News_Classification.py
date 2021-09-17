import pandas as pd                                             #Pandas library allows us to process and analyse extensive data with ease, using special data structures and functions
                                                                #Scikit-learn or sklearn is an important ML package for Python.
from sklearn.model_selection import train_test_split            #Import the train_test_split function, which splits the labelled dataset into a random training and testing dataset to allow cross validation
from sklearn.feature_extraction.text import TfidfVectorizer     #Import the TfidfVectorizer which converts the dataset into a matrix of TF-IDF features
from sklearn.linear_model import PassiveAggressiveClassifier    #Import the PassiveAggressiveClassifier which is an online learning classification algorithm
from sklearn.metrics import accuracy_score                      #Import the accuracy_score function which provides a metric as to the accuracy of the algorithm

df=pd.read_csv('C:\\news.csv')                                                                          #Read the training dataset
df.shape                                                                                                #Get shape and head
labels=df.label                                                                                         #Get the labels
runs = 43                                                                                               #Number of iterations to run on the data
avg_acc = 0
avg_fake = 0

for rs in range(runs):
    x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=rs)  #Split the dataset
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)                                  #Initialize a TfidfVectorizer
    tfidf_train=tfidf_vectorizer.fit_transform(x_train)                                                 #Fit and transform train set
    tfidf_test=tfidf_vectorizer.transform(x_test)                                                       #Transform test set
    pac=PassiveAggressiveClassifier(max_iter=50)                                                        #Initialize a PassiveAggressiveClassifier
    pac.fit(tfidf_train,y_train)                                                                        #Fit the classifier on the transformed training set and the original set
    pred=pac.predict(tfidf_test)                                                                        #Predict on the test set
    score=accuracy_score(y_test,pred)*100                                                               #Calculate the accuracy                                                                 
    avg_acc += score/runs                                                                               
    df2=pd.read_csv('GRD.csv')                                                                          #Read the dataset to be analysed
    tfidf_pred=tfidf_vectorizer.transform(df2['text'])                                                  #Transform the final dataset
    pred=pac.predict(tfidf_pred)                                                                        #Predict on the final set
    fin = list(pred)                                                                                    #Save the predictions as a list
    fake = ((fin.count("FAKE")/len(fin))*100)                                                           #Calculate the percentage of fake news
    avg_fake += fake/runs
print("Average accuracy:",avg_acc,"%")                                                                  #Print the average accuracy of the classifier
print("Percentage of fake news:",avg_fake,"%")                                                          #Print the percentage of fake news on average

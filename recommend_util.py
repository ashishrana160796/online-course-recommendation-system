# Decided Functionality: Takes course-id (String) as input in the form of a function, 
# returns a list of recommended course-ids.


# Procedure for handling inputs:

# 1. Load the earlier traied clusteriazation model. Model, already trained. Don't train it again.
# 2. Give cluster labels/categories to all 8-clusters formed. A mapping like procedure, done manually here.
# Note:
# Limitaion: Categories with less courses got diluted down on current clusters that are formed.
# Mitigation: Another possible algorithm like SVD or add more cluster to be able to detect those cluster.
# 3. Assign labels to all live courses that are not retired & store them in a data-frame.
# Advantage: These labels are dynamic & dependent on model. Hence, with better models better labels can get assigned.
# 4. Receive input from user in terms of string, predict its cluster-category.
# 5. Recommend 10[or 'n'] random courses of same category based on the given input provided & predicted by user.




# import statements
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


# 1. load model and previous preprocessing.

# load model only once
with open('finalized_model.sav', 'rb') as fid:
    model = pickle.load(fid)

# X = vectorizer.fit_transform(course_df['InputString'])
# This will give an error as incorrect number of features, i.e. if features from a different data-frame is used
# seperate code snippet for building vocabulary for trained model
courses_df = pd.read_csv("data/courses.csv")
courses_df = courses_df.dropna(how='any')
courses_df['Description'] = courses_df['Description'].replace({"'ll": " "}, regex=True)
courses_df['CourseId'] = courses_df['CourseId'].replace({"-": " "}, regex=True)
comb_frame = courses_df.CourseId.str.cat(" "+courses_df.CourseTitle.str.cat(" "+courses_df.Description))
comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)

# Add clustering labels to every non-retired course
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame)
    

# 2. Current utility variable and data frame preprocessing

# Verbose only, Not getting used in the code: creating labels for clusters manually
label_dict = {
                 0: "Gaming Professionals",
                 1: "Manufacturing & Design",
                 2: "Software Development",
                 3: "Data Professionals",
                 4: "Information & Cyber Security",
                 5: "Movie Making & Animation",
                 6: "IT Ops",
                 7: "Graphic Design"
            }
    
# load the complete data in a dataframe
course_df = pd.read_csv("data/courses.csv")
# drop retired course from analysis. But, courses with no descriptions are kept.
course_df = course_df[course_df.IsCourseRetired == 'no']
    
# create new column in dataframe which is combination of (CourseId, CourseTitle, Description) in existing data-frame
course_df['InputString'] = course_df.CourseId.str.cat(" "+course_df.CourseTitle.str.cat(" "+course_df.Description))

course_df['ClusterPrediction'] = ""


def cluster_predict(str_input):
    Y = vectorizer.transform(list(str_input))
    prediction = model.predict(Y)
    return prediction

# Cluster category for each live course
course_df['ClusterPrediction']=course_df.apply(lambda x: cluster_predict(course_df['InputString']), axis=0)


def recommend_util(str_input):
    
    # Predict category of input string category
    temp_df = course_df.loc[course_df['CourseId'] == str_input]
    temp_df['InputString'] = temp_df.CourseId.str.cat(" "+temp_df.CourseTitle.str.cat(" "+temp_df['Description']))
    str_input = list(temp_df['InputString'])
    
    prediction_inp = cluster_predict(str_input)
    prediction_inp = int(prediction_inp)
    
    temp_df = course_df.loc[course_df['ClusterPrediction'] == prediction_inp]
    temp_df = temp_df.sample(10)
    
    return list(temp_df['CourseId'])

if __name__ == '__main__':
    queries = ['wp7-core', 'ef41-data-access', 'nosql-big-pic', 'procedural-ice-modeling-softimage-153', \
               'beginners-guide-shading-networks-softimage-510', 'centralized-logging-elastic-stack', \
               'apache-pig-data-transformations']

    for query in queries:
        res = recommend_util(query)
        print(res)

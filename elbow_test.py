# Elbow Method for finding right number of clusters
# that minimizes SSE (Sum of squared errors).

# 1. import statments
import numpy as np                                                      
import pandas as pd                                                    
import matplotlib.pyplot as plt                                        
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer



# 2. Ingestion & Preprocessing steps 

# read data, from source
course_df = pd.read_csv("data/courses.csv")

# drop rows with NaN values for any column, specifically 'Description'
course_df = course_df.dropna(how='any')

# Preprocess description for models
# Remove stopwards, numbers(carries information about courses don't remove them.)
# Remove 'll' ASAP. Anything ending with " 'll " has to be replaced.
# Try removing extra text, keep important phrases & nouns

# Pre-preprocessing step: remove words like we'll, you'll, they'll etc.
course_df['Description'] = course_df['Description'].replace({"'ll": " "}, regex=True)
course_df['CourseId'] = course_df['CourseId'].replace({"-": " "}, regex=True)

# Combine three columns namely: CourseId, CourseTitle, Description
# As all of them reveal some information about the course
comb_frame = course_df.CourseId.str.cat(" "+course_df.CourseTitle.str.cat(" "+course_df.Description))

# remove all characters except numbers and alphabets
comb_frame = comb_frame.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)


# 3. Simulating elbow-test to arrive at optimum number of clusters 

# Create word vectors from combined frames 
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame)

# data-structure to store Sum-Of-Square-Errors
sse = {}

# Looping over multiple values of k from 1 to 30
for k in range(1, 31):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100).fit(X)
    comb_frame["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_


# Sample Values obtained from simulation, directly can be inputted to create a quick sample plot.
"""
# Sum of Square values obtained from first 30 iterations with 'k' ranging from 1 to 30.

sse= {1: 7871.172561995738, \
 2: 7801.457204502822, \
 3: 7753.244659579704, \
 4: 7703.298754909891, \
 5: 7665.787975547076, \
 6: 7623.323536225195, \
 7: 7591.5728741240055, \
 8: 7558.591511379371, \
 9: 7524.354289929809, \
 10: 7495.400302680042, \
 11: 7475.148244309455, \
 12: 7446.897970324125, \
 13: 7429.919142273945, \
 14: 7411.2034983449685, \
 15: 7374.572369199682, \
 16: 7354.990461452205, \
 17: 7341.887620426839, \
 18: 7325.447310490266, \
 19: 7301.089847503103, \
 20: 7277.112921936699, \
 21: 7265.311767795575, \
 22: 7243.697707006134, \
 23: 7224.009806461437, \
 24: 7208.6061482901505, \
 25: 7211.596618429591, \
 26: 7186.979156239425, \
 27: 7162.4230578913875, \
 28: 7146.463029004067, \
 29: 7131.118902368547, \
 30: 7127.340534336784}
"""

# Plotting the curve with 'k'-value vs SSE
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
# Save the Plot in current directory
plt.savefig('elbow_method.png')

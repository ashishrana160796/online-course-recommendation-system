# Working of this recommendation system pipeline.

# 1. import statments
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import Normalizer
import pickle

# 2. Ingest data from source & Preprocessing that needs to be carried out.

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


# 3. Train model with Latent semantic analysis with SVD & k-means

# Create word vectors from combined frames 
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(comb_frame)

# true_k, derived from elbow metho and confirmed from pluralsight's website
# true_k = 8
# Instead, k=30 with elbow is picked which is producing lower error
true_k = 30

# usig SVD for LSA
# svd = TruncatedSVD(true_k)
# lsa = make_pipeline(svd, Normalizer(copy=False))
# X = lsa.fit_transform(X)

# Running model with 15 different centroid initializations & maximum iterations are 500
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=15)
model.fit(X)


# 5. Preview clusters, test your model for demo and save your model for further use.

# Preview top 15 words in each cluster, and accordingly different clusters can be assigned 
# a given categories out of 8 categories on pluralsight's website.

# Create a hashmap, mapping each cluster to a given category.

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind]),
    print


# For testing which cluster the following course is having, manipulate string as 'CourseId(with string replacement from '-' to a blankspace)+" "+CourseTitle+" "+Description'

# Y = vectorizer.transform(["aspdotnet data ASP.NET 3.5 Working With Data ASP.NET has established itself as one of the most productive environments for building web applications and more developers are switching over every day. The 2.0 release of ASP.NET builds on the same componentry of 1.1, improving productivity of developers even further by providing standard implementations of common Web application features like membership, persistent user profile, and Web parts, among others. The 3.5 release adds several new controls including the flexible ListView and the LinqDataSource, as well as integrated suport for ASP.NET Ajax. This course will cover the data access, caching, and state management features of ASP.NET."])
# prediction = model.predict(Y)
# print(prediction)     # A cluster category will be given as an output.


# Save machine learning model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# For, loading the final model

# with open('finalized_model.sav', 'rb') as fid:
#     model = pickle.load(fid)


# References Links Used or Studied

# 1. https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
# 2. https://www.pluralsight.com/browse, For finding the original number of categories as '8' for all courses.
# 3. https://stats.stackexchange.com/questions/155880/text-mining-how-to-cluster-texts-e-g-news-articles-with-artificial-intellige, A possible solution improvement from this thread. First try the svd approach[Currently, using similar to this approach only] and then doc2vec approach.
# 4. https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion, Elbow method to find original value of 'k'.
# 5. https://pythonprogramminglanguage.com/kmeans-text-clustering/, Key resource in-use for clustering.

# Limitations and Analysis of (code & data)

# 1. Data limited only to course descriptions only, with few containing NaN values.
# 2. Even, retired courses are used for training but they are not displayed as recommended courses. Around 900 courses are retired and they can't be recommended.
# 3. 


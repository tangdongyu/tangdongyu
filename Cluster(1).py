from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,SpectralClustering,AgglomerativeClustering,DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer


#读取文件并对文件进行处理

def Preprocessing():

	f = open('E:/pycharmworker/worker/Tweets.txt','r')
	text = []
	clusters = []

	for line in f.readlines():
		words = eval(line)
		text.append(words['text'])
		clusters.append(words['cluster'])
	tfidfvectorizer = TfidfVectorizer(stop_words='english')
	X = tfidfvectorizer.fit_transform(text).toarray()
	return X,clusters



#统计文本中类的个数

def count_cluster(clusters):
	count = 0
	temp = []
	for cluster in clusters:
		if (cluster not in temp):
			temp.append(cluster)
			count = count + 1
	return (count)


#K-means
def K_means(X,y,k):
	print(1)
	y_pred = KMeans(n_clusters = k,random_state = 10).fit_predict(X)
	print('K-means:', normalized_mutual_info_score(y_pred, y))


#Affinity propagation
def Affinity_propagation(X,y):

	print(2)
	y_pred = AffinityPropagation(damping = 0.6).fit_predict(X)
	print('Affinity propagation:', normalized_mutual_info_score(y_pred, y))

#Mean-Shift
def Mean_Shift(X,y):
	print(3)
	y_pred = MeanShift(bandwidth = 0.6,bin_seeding = True).fit_predict(X)
	print('Mean-Shift:', normalized_mutual_info_score(y_pred, y))

#Spectral Clustering
def Spectral_Clustering(X,y,k):

	print(4)
	y_pred = SpectralClustering(n_clusters = k,eigen_solver="arpack").fit_predict(X)
	print('Spectral Clustering:', normalized_mutual_info_score(y_pred, y))

#Agglomerative Clustering
def Agglomerative_Clustering(X,y,k):

	print(5)
	y_pred = AgglomerativeClustering(n_clusters = k,affinity = 'euclidean',linkage = 'ward').fit_predict(X)
	print('Agglomerative Clustering:', normalized_mutual_info_score(y_pred, y))


#DBSCAN算法

def DBSCAN_(X,y):
	print(6)
	y_pred = DBSCAN(eps = 1,min_samples = 1).fit_predict(X)
	print('DBSCAN:', normalized_mutual_info_score(y_pred, y))

#Gaussian_Mixture
def Gaussian_Mixture(X,y,k):
	print(7)
	y_pred = GaussianMixture(n_components = k,covariance_type = 'diag',random_state = 10).fit_predict(X)
	print('Gaussian Mixture:', normalized_mutual_info_score(y_pred, y))


if __name__ == '__main__':

	X,clusters = Preprocessing()
	k = count_cluster(clusters)
	K_means(X,clusters,k)
	Affinity_propagation(X,clusters)
	Mean_Shift(X,clusters)
	Spectral_Clustering(X,clusters,k)
	Agglomerative_Clustering(X,clusters,k)
	DBSCAN_(X,clusters)
	Gaussian_Mixture(X,clusters,k)
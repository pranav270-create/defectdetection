import numpy as np

def kmeans(x, K, niter, seed=123):
    """
    x: array of shape (N, D)
    K: integer
    niter: integer

    centroids: array of shape (K, D)
    labels: array of shape (height*width, )
    """
    np.random.seed(seed)
    idx = np.random.choice(len(x), K, replace=False)

    # Randomly choose centroids
    centroids = x[idx, :]

    # Initialize labels
    labels = np.zeros((x.shape[0], ))
    changed = True
    i = 0
    
    while changed:
        
        if i == niter - 1:
            changed = False
            
        X_norms = np.array([np.sum(np.square(x),axis=1)])
        # Square of the matrix elements and summing across the rows
        centroids_norms = np.array([np.sum(np.square(centroids),axis=1)])
        # Finding xi*wj by doing X*WT
        X_dot_centroids = 2*np.matmul(x,np.transpose(centroids))

        distances = np.transpose(X_norms) + centroids_norms - X_dot_centroids
        temp = np.argmin(distances, axis = 1)
        
        if np.array_equal(temp, labels):
            changed = False 
        
        labels = temp
        centroid_labels, centroid_counts = np.unique(labels, return_counts=True)
        indx = np.isin(np.arange(K), centroid_labels)
        # number of times each centroid is associated with a point
        counts = np.zeros(K)
        # The index of the count matrix corresponds to the number of times the entry appears in u2
        counts[indx] = centroid_counts[indx]
        # adding 1e-20 to avoid division by 0 in final line
        counts = counts.reshape(K, 1) + 1e-20 
        # vectorized way of calculating sum over rows associated with each cluster
        sums = (np.eye(K)[labels]).T @ x
        centroids = sums/counts
        
        i+=1
    
    return labels, centroids

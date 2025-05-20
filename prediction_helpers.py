import numpy as np


def cosine_sim(mat1, mat2):
    """Calculate cosine similarity between two matrices.
    
    Args:
        mat1: First matrix of shape (n_samples, n_features)
        mat2: Second matrix of shape (n_samples, n_features)
        
    Returns:
        Array of cosine similarities for each pair of samples
    """
    prod = mat1 * mat2
    normf = lambda x: np.sqrt(np.sum(x**2, axis=1))
    normx, normy = normf(mat1), normf(mat2)
    # Avoid division by zero
    normx = np.maximum(normx, 1e-10)
    normy = np.maximum(normy, 1e-10)
    return np.sum(prod, axis=1) / (normx * normy)


def manhattan_sim(mat1, mat2):
    """Calculate Manhattan similarity between two matrices.
    
    Args:
        mat1: First matrix of shape (n_samples, n_features)
        mat2: Second matrix of shape (n_samples, n_features)
        
    Returns:
        Array of Manhattan similarities for each pair of samples
    """
    diffs = np.sum(np.abs(mat1 - mat2), axis=1)
    return 1 - diffs


def get_preds(xsent_encoded, ysent_encoded, globalsim=cosine_sim, subsim=cosine_sim, biases=None, n=15, dim=16):
    """Get predictions of model for paired sentence vectors.
    
    Args:
        xsent_encoded: Matrix of sentence vectors for first sentences
        ysent_encoded: Matrix of sentence vectors for second sentences
        globalsim: Function to compute global similarity
        subsim: Function to compute sub-embedding similarity
        biases: Score bias coefficients for metrics (optional)
        n: Number of metrics that are modeled (besides residual)
        dim: Feature dimension
        
    Returns:
        Matrix with predictions for each pair and each feature
    """
    if biases is None:
        biases = np.ones(n)
    
    # Global SBERT similarities
    simsglobal = globalsim(xsent_encoded, ysent_encoded)
    
    # Residual similarities
    simsresidual = globalsim(xsent_encoded[:, dim*n:], ysent_encoded[:, dim*n:])
    
    # Extract feature matrices for each metric
    metric_features = []
    for i in range(0, dim*n, dim):
        metric_features.append((xsent_encoded[:, i:i+dim], ysent_encoded[:, i:i+dim]))
    
    # Calculate similarities for each metric
    metric_sims = []
    for i in range(n):
        xfea = metric_features[i][0]
        yfea = metric_features[i][1]
        simfea = subsim(xfea, yfea)
        metric_sims.append(simfea)
    
    # Apply biases and transpose
    metric_sims = np.array(metric_sims) * biases[:, np.newaxis]
    metric_sims = metric_sims.T
    
    # Combine all similarities
    preds = np.concatenate((simsglobal[:, np.newaxis], metric_sims, simsresidual[:, np.newaxis]), axis=1)
    return preds
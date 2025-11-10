PROMPT = """ Given a dataset of 3D points, available in the `python_expression` tool as the variable `dataset`.
         Your goal is to group these points into {num_clusters} coherent clusters based on their similarity.
         1.  First, normalize the data. For each of the three features (dimensions), subtract the feature's mean and divide by its standard deviation.
         2.  Using the normalized data:
               a. Randomly initialize {num_clusters} cluster centroids.
               b. For {num_iterations} iterations:
                  i. Assign each point to the nearest centroid based on Euclidean distance.
                  ii. Recalculate the centroids based on the new cluster assignments.
         3.  After completing the iterations, submit a list of [point, cluster_id] pairs where each point from the **original, unnormalized** dataset is paired with its final cluster index.
         
      Each point should be a string in the form "(x, y, z)" and cluster_id should be an integer from 0 to {num_clusters}-1.
      The list should contain exactly the same number of entries as the original dataset, preserving all points including duplicates.
      Format: [["(x, y, z)", cluster_id], ["(x, y, z)", cluster_id], ...]
      Use the `python_expression` tool for all calculations.
      Avoid generating unnecessary output such as code comments and do not print explanations or intermediate reasoning.**
"""

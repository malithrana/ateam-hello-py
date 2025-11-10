import json
import numpy as np

# Grading function
def kmeans_expected_checker(answer, num_clusters, points_data):
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError:
            print("Failed to parse answer string as JSON.")
            return False

    if not isinstance(answer, list):
        print("Answer is not a list.")
        return False

    # Check length matches
    if len(answer) != len(points_data):
        print(
            f"Answer length {len(answer)} does not match dataset length {len(points_data)}."
        )
        return False

    parsed_answer = []
    for item in answer:
        try:
            if not isinstance(item, list) or len(item) != 2:
                print(f"Invalid item format: {item}. Expected [point_str, cluster_id].")
                return False

            point_str, cluster_id = item
            # More robust parsing to handle potential formatting inconsistencies
            point_tuple = tuple(
                map(int, point_str.strip("()").replace(" ", "").split(","))
            )

            if cluster_id not in range(num_clusters):
                print(f"Invalid cluster ID: {cluster_id}.")
                return False

            parsed_answer.append((point_tuple, cluster_id))
        except (ValueError, AttributeError, TypeError) as e:
            print(f"Failed to parse item: {item}. Error: {e}")
            return False

    # Extract points and cluster IDs from parsed answer
    parsed_points = [point for point, _ in parsed_answer]
    cluster_ids = np.array([cluster_id for _, cluster_id in parsed_answer])

    # Check that parsed points match original dataset exactly (including order and duplicates)
    if parsed_points != points_data:
        print("Submitted points do not match original dataset.")
        return False

    # Check all clusters are represented
    if set(cluster_ids) != set(range(num_clusters)):
        print("Not all clusters have points assigned.")
        return False

    # Perform the correct data normalization to check the agent's work
    points_array = np.array(points_data)
    mean = np.mean(points_array, axis=0)
    std = np.std(points_array, axis=0)
    normalized_points = (points_array - mean) / std

    # Calculate centroids for each cluster using normalized data
    normalized_centroids = np.array(
        [
            np.mean(normalized_points[cluster_ids == i], axis=0)
            for i in range(num_clusters)
        ]
    )

    # Verify each point is assigned to its nearest centroid
    distances = np.linalg.norm(
        normalized_points[:, np.newaxis, :] - normalized_centroids[np.newaxis, :, :],
        axis=2,
    )
    closest_centroids = np.argmin(distances, axis=1)

    if not np.all(closest_centroids == cluster_ids):
        print("Some points are not assigned to their nearest centroid.")
        return False

    return True

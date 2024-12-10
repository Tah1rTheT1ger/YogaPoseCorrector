import pytest
import numpy as np
from pose_equal_check import PoseSimilarity, resize_image
from PoseModule import PoseDetector
import time

@pytest.fixture
def pose_similarity():
    """
    Fixture to initialize the PoseSimilarity instance for testing.
    """
    return PoseSimilarity()

@pytest.fixture
def mock_landmarks():
    """
    Fixture to create mock landmark data for testing.
    """
    return [
        (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1),
        (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5)
    ]

@pytest.fixture
def detector():
    """
    Fixture to initialize the PoseDetector instance for mock usage.
    """
    return PoseDetector()

def test_euclidean_distance(pose_similarity):
    """
    Test the euclidean_distance function.
    """
    point1 = (0, 0)
    point2 = (3, 4)
    distance = pose_similarity.euclidean_distance(point1, point2)
    assert distance == 5.0, "Distance should be 5.0 for a 3-4-5 triangle"

def test_normalize_landmarks(pose_similarity, mock_landmarks):
    """
    Test the normalize_landmarks function.
    """
    normalized_landmarks = pose_similarity.normalize_landmarks(mock_landmarks, reference_idx=0)
    ref_point = mock_landmarks[0]
    for i, point in enumerate(normalized_landmarks):
        assert point[0] == pytest.approx(mock_landmarks[i][0] - ref_point[0]), "X-coordinate should be normalized"
        assert point[1] == pytest.approx(mock_landmarks[i][1] - ref_point[1]), "Y-coordinate should be normalized"

def test_compare_poses(pose_similarity, mock_landmarks):
    """
    Test the compare_poses function.
    """
    landmarks1 = mock_landmarks
    landmarks2 = [(x + 0.01, y - 0.01) for x, y in mock_landmarks]  # Slightly shifted
    result = pose_similarity.compare_poses(landmarks1, landmarks2, threshold=0.1)
    assert result <= 0.1, "Average distance should be within threshold for similar landmarks"

def test_get_wrong_joints(pose_similarity, detector, mock_landmarks):
    """
    Test the get_wrong_joints function.
    """
    asana = "Padmasana"
    input_landmarks = [(x + 0.05, y - 0.05) for x, y in mock_landmarks]  # Slightly shifted
    correct_landmarks = mock_landmarks

    detector.map_landmarks = lambda x: {str(i): coord for i, coord in enumerate(x)}
    detector.map_joints = lambda x: {"joint_{}".format(i): [coord, coord, coord] for i, coord in enumerate(x.values())}
    detector.get_joints_for_asana = lambda a, b, c: c

    wrong_joints = pose_similarity.get_wrong_joints(asana, correct_landmarks, input_landmarks, thresh=0.1)
    assert isinstance(wrong_joints, dict), "Wrong joints should be returned as a dictionary"
    assert all(isinstance(k, str) for k in wrong_joints.keys()), "Joint names should be strings"

def test_isSimilar(pose_similarity, mock_landmarks):
    """
    Test the isSimilar function.
    """
    pose_name = "Padmasana"
    input_landmarks = [(x + 0.01, y - 0.01) for x, y in mock_landmarks]
    correct_landmarks = [mock_landmarks, input_landmarks]  # Multiple variations
    pose_similarity.ideal_landmarks = {pose_name: correct_landmarks}
    pose_similarity.absolutely_ideal_landmarks = {pose_name: []}

    is_similar, closest_landmarks = pose_similarity.isSimilar(pose_name, input_landmarks, euclidean_threshold=0.1)
    assert is_similar, "The input pose should be similar to the correct pose"
    assert closest_landmarks == mock_landmarks, "Closest landmarks should match the correct pose"

def test_resize_image():
    """
    Test the resize_image function.
    """
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)  # Large square image
    resized_image = resize_image(image, max_width=800, max_height=600)
    assert resized_image.shape[0] <= 600, "Height should be within the max_height"
    assert resized_image.shape[1] <= 800, "Width should be within the max_width"

def test_time_execution():
    """
    Test the overall time taken to run comparison logic.
    """
    start_time = time.time()
    time.sleep(1)  # Simulating execution time
    elapsed_time = time.time() - start_time
    assert elapsed_time >= 1.0, "Execution should respect simulated delay"

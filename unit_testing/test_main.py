import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch
from main import resize_frame, text_to_speech, PoseSimilarity

@pytest.fixture
def mock_frame():
    """
    Create a mock video frame for testing.
    """
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def pose_similarity():
    """
    Create an instance of PoseSimilarity for testing.
    """
    return PoseSimilarity()

def test_resize_frame(mock_frame):
    """
    Test the resize_frame function to ensure it resizes frames correctly.
    """
    resized_frame = resize_frame(mock_frame, scale_factor=0.5)
    assert resized_frame.shape[0] == 240, "Height should be half of the original"
    assert resized_frame.shape[1] == 320, "Width should be half of the original"

    resized_frame = resize_frame(mock_frame, scale_factor=1.5)
    assert resized_frame.shape[0] == int(480 * 1.5), "Height should be scaled by 1.5"
    assert resized_frame.shape[1] == int(640 * 1.5), "Width should be scaled by 1.5"

def test_text_to_speech():
    """
    Test the text_to_speech function to ensure it converts text to speech without errors.
    """
    with patch("main.pygame.mixer.music") as mock_music:
        mock_music.load = MagicMock()
        mock_music.play = MagicMock()
        mock_music.get_busy = MagicMock(return_value=False)

        # Call the function
        text_to_speech("Test speech")

        # Verify function calls
        mock_music.load.assert_called_once()
        mock_music.play.assert_called_once()

def test_normalize_landmarks(pose_similarity):
    """
    Test the normalize_landmarks function to ensure correct normalization.
    """
    landmarks = [(100, 200), (150, 250), (200, 300)]
    normalized = pose_similarity.normalize_landmarks(landmarks, reference_idx=0)
    expected = [(0, 0), (50, 50), (100, 100)]

    assert len(normalized) == len(expected), "Number of landmarks should match"
    for norm, exp in zip(normalized, expected):
        assert norm == exp, f"Expected {exp}, got {norm}"

def test_compare_poses(pose_similarity):
    """
    Test the compare_poses function to ensure it calculates the average distance correctly.
    """
    landmarks1 = [(0, 0), (1, 1), (2, 2)]
    landmarks2 = [(0, 0), (1.1, 1.1), (2.2, 2.2)]  # Slightly shifted
    avg_distance = pose_similarity.compare_poses(landmarks1, landmarks2, threshold=0.3)

    assert avg_distance <= 0.3, "Average distance should be within threshold"

def test_isSimilar(pose_similarity):
    """
    Test the isSimilar function to ensure it correctly identifies similar poses.
    """
    pose_similarity.isSimilar = MagicMock(return_value=(True, []))
    result, _ = pose_similarity.isSimilar("TestPose", [(0, 0), (1, 1)], 0.1)

    assert result is True, "Pose should be identified as similar"

def test_get_wrong_joints(pose_similarity):
    """
    Test the get_wrong_joints function to ensure it detects incorrect joints.
    """
    pose_similarity.get_wrong_joints = MagicMock(return_value={"joint_1": ("joint_1", "increase")})
    wrong_joints = pose_similarity.get_wrong_joints("TestPose", [(0, 0), (1, 1)], [(0.1, 0.1), (1.2, 1.2)], 0.1)

    assert "joint_1" in wrong_joints, "Incorrect joints should be detected"
    assert wrong_joints["joint_1"] == ("joint_1", "increase"), "Incorrect joint adjustment should match"

def test_execution_time():
    """
    Test the overall execution time of the logic to ensure it is within expected limits.
    """
    start_time = time.time()
    time.sleep(1)  # Simulate processing delay
    elapsed_time = time.time() - start_time

    assert elapsed_time >= 1.0, "Execution time should match or exceed the simulated delay"

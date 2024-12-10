import pytest
import numpy as np
import cv2 as cv
from PoseModule import PoseDetector

@pytest.fixture
def pose_detector():
    """Fixture to initialize the PoseDetector instance for testing."""
    return PoseDetector()

@pytest.fixture
def mock_frame():
    """Fixture to create a mock video frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_landmarks():
    """Fixture to create mock landmark data for testing."""
    return [
        [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], 
        [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], 
        [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [0.1, 0.9], 
        [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], 
        [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [0.1, 0.9], [0.2, 0.8], 
        [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], 
        [0.8, 0.2], [0.9, 0.1]
    ]

def test_findPose(pose_detector, mock_frame):
    """
    Test the findPose method to ensure it processes the frame without errors
    and returns a valid frame.
    """
    output_frame = pose_detector.findPose(mock_frame, draw=False)
    assert isinstance(output_frame, np.ndarray), "Output should be a numpy array"
    assert output_frame.shape == mock_frame.shape, "Output frame dimensions should match input"

def test_findPosition_no_landmarks(pose_detector, mock_frame):
    """
    Test the findPosition method when no landmarks are detected.
    """
    pose_detector.results = None  # Simulate no detection
    landmarks = pose_detector.findPosition(mock_frame)
    assert landmarks == [], "Landmarks list should be empty when no landmarks are detected"

def test_findPosition_with_landmarks(pose_detector, mock_frame, mock_landmarks):
    """
    Test the findPosition method when landmarks are detected.
    """
    # Simulate detection with mock landmarks
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    class MockResults:
        def __init__(self, landmarks):
            self.pose_landmarks = landmarks

    # Create a list of MockLandmark objects
    landmarks_list = [MockLandmark(x, y) for x, y in mock_landmarks]
    mock_pose_landmarks = MockResults(landmarks_list)

    # Assign mock landmarks to the pose detector
    pose_detector.results = mock_pose_landmarks

    # Test the findPosition function
    landmarks = pose_detector.findPosition(mock_frame)
    assert len(landmarks) == len(mock_landmarks), "Number of landmarks should match mock data"

def test_map_landmarks(pose_detector, mock_landmarks):
    """
    Test the map_landmarks method to ensure correct mapping of body parts to landmarks.
    """
    landmark_dict = pose_detector.map_landmarks(mock_landmarks)
    assert 'nose' in landmark_dict, "'nose' key should exist in the landmark dictionary"
    assert 'left_eye' in landmark_dict, "'left_eye' key should exist in the landmark dictionary"
    assert isinstance(landmark_dict['nose'], list), "Mapped value for 'nose' should be a list"
    assert len(landmark_dict) == 34, "Landmark dictionary should have 34 keys"

def test_calculate_angle(pose_detector):
    """
    Test the calculate_angle method to ensure correct calculation of angles.
    """
    # Test with a right angle triangle
    points = [(1, 1), (0, 0), (1, 0)]  # 90 degrees at (0, 0)
    angle = pose_detector.calculate_angle(points)
    assert pytest.approx(angle, 0.1) == 90.0, "Angle should be approximately 90 degrees"

    # Test with a straight line
    points = [(1, 0), (0, 0), (-1, 0)]  # 180 degrees at (0, 0)
    angle = pose_detector.calculate_angle(points)
    assert pytest.approx(angle, 0.1) == 180.0, "Angle should be approximately 180 degrees"

    # Test with invalid input
    points = []
    angle = pose_detector.calculate_angle(points)
    assert angle is None, "Angle should be None for invalid input"

def test_ChangeColor(pose_detector, mock_frame):
    """
    Test the ChangeColor method to ensure the pose connections are updated with the specified color.
    """
    output_frame = pose_detector.ChangeColor(mock_frame, color=(255, 0, 0), draw=True)
    assert isinstance(output_frame, np.ndarray), "Output should be a numpy array"
    assert output_frame.shape == mock_frame.shape, "Output frame dimensions should match input"


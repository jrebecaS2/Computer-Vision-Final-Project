import cv2
import numpy as np
from sklearn.decomposition import PCA


MAX_KEYPOINTS = 100
MIN_TRACKED_POINTS = 40
FIXED_POINTS = 40
CALIBRATION_TIME_SEC = 5
DEVIATION_THRESHOLD = 4 
ROLLING_BUFFER_SIZE = 30

BLUR_KERNEL = 5  
CANNY_LOW = 80  
CANNY_HIGH = 200  
CROP_RATIO = 0.75  
THRESHOLD_VALUE = 127  

def preprocess_frame(frame):
    """
    Preprocess frame to isolate upper-body region and edge map.
    Pipeline: Gaussian blur Canny edge detection upper-body crop binary threshold.
    
    Returns:
        edge_frame: Edge map of isolated upper-body region
        mask: Binary mask where body edges are white (255), background is black (0)
    """
    #gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    #canny edge detection
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    h, w = edges.shape
    crop_h = int(h * CROP_RATIO)
    edges_cropped = edges[:crop_h, :]
    
    #binary threshold to isolate dominant body blob
    _, binary_mask = cv2.threshold(edges_cropped, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    
    return edges_cropped, binary_mask


def detect_keypoints(frame):
    """
    Detect Shi-Tomasi keypoints on preprocessed upper-body region.
    """
    edges, mask = preprocess_frame(frame)
    
    # shi-Tomasi corner detection
    points = cv2.goodFeaturesToTrack(
        mask,
        maxCorners=MAX_KEYPOINTS,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7
    )
    return points


def track_keypoints(prev_frame, curr_frame, prev_points):
    """
    Track keypoints using KLT optical flow.
    Returns:
        good_new: Nx2 array of successfully tracked new points
        good_old: Nx2 array of corresponding old points
    """
    if prev_points is None or len(prev_points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    if next_points is None or status is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    status = status.reshape(-1)
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    if len(good_new) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    return good_new.reshape(-1, 2), good_old.reshape(-1, 2)


def compute_feature_vector(points, fixed_points=FIXED_POINTS):
    """
    Given Nx2 array of keypoints, compute a centroid-normalized,
    fixed-length flattened feature vector.

    If there are fewer than fixed_points points, return None.
    """
    if points is None or len(points) < fixed_points:
        return None

    # Use a fixed number of points so every feature vector has the same shape
    points = points[:fixed_points]

    centroid = np.mean(points, axis=0)
    normed = points - centroid

    # Optional scale normalization so moving slightly closer/farther affects less
    scale = np.linalg.norm(normed, axis=1).mean()
    if scale > 1e-6:
        normed = normed / scale

    return normed.flatten()


def calibrate_pca(feature_list):
    """
    Fit PCA to a list of feature vectors. 
    Returns: mean_vec, pca, baseline_deviation (mean error of training data)
    
    baseline_deviation is the average PCA reconstruction error on training data.
    Used later to compute RELATIVE deviation from the calibration baseline.
    """
    valid = [f for f in feature_list if f is not None]
    if len(valid) == 0:
        raise ValueError("No valid calibration features collected.")

    # Keep only vectors that match the first valid shape
    first_shape = valid[0].shape
    valid = [f for f in valid if f.shape == first_shape]

    if len(valid) == 0:
        raise ValueError("Calibration features do not share a common shape.")

    X = np.stack(valid)

    n_components = min(10, X.shape[0], X.shape[1])
    if n_components < 1:
        raise ValueError("Not enough data to fit PCA.")

    pca = PCA(n_components=n_components)
    pca.fit(X)

    mean_vec = np.mean(X, axis=0)
    
    baseline_errors = []
    for feature in valid:
        proj = pca.transform([feature])
        recon = pca.inverse_transform(proj)[0]
        error = np.linalg.norm(feature - recon)
        baseline_errors.append(error)
    
    baseline_deviation = np.mean(baseline_errors)
    
    return mean_vec, pca, baseline_deviation


def compute_deviation(feature_vector, mean_vec, pca, baseline_deviation):
    """
    Project feature_vector into PCA space, reconstruct, and compute RELATIVE L2 error.
    
    Returns the deviation relative to the calibration baseline.
    - 0 or negative: same as calibration posture (good)
    - Positive and large: different from calibration (bad)
    """
    if feature_vector is None or mean_vec is None or pca is None or baseline_deviation is None:
        return 0.0

    # Safety check: feature shape must match the calibration shape
    if feature_vector.shape != mean_vec.shape:
        return 0.0

    proj = pca.transform([feature_vector])
    recon = pca.inverse_transform(proj)[0]
    error = np.linalg.norm(feature_vector - recon)
    
    relative_error = error - baseline_deviation
    
    return float(relative_error)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_frame = None
    prev_points = None

    calibration_features = []
    calibration_start = None
    calibrated = False
    mean_vec = None
    pca = None
    baseline_deviation = None

    deviation_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27: 
            break

        if not calibrated:
            if prev_points is None or len(prev_points) < MIN_TRACKED_POINTS:
                prev_points = detect_keypoints(frame)
                prev_frame = frame.copy()

                if calibration_start is None:
                    calibration_start = cv2.getTickCount() / cv2.getTickFrequency()

                cv2.putText(
                    frame,
                    "Calibrating... detecting points",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("Posture Analyzer", frame)
                continue

            #tracking 
            new_points, _ = track_keypoints(prev_frame, frame, prev_points)
            prev_frame = frame.copy()

            if len(new_points) < MIN_TRACKED_POINTS:
                prev_points = None
                cv2.putText(
                    frame,
                    "Calibrating... re-detecting points",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("Posture Analyzer", frame)
                continue

            prev_points = new_points.reshape(-1, 1, 2)

            # feature vector
            feature_vec = compute_feature_vector(new_points)
            if feature_vec is not None:
                calibration_features.append(feature_vec)

            #draw keypoints
            for pt in new_points:
                cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

            elapsed = (cv2.getTickCount() / cv2.getTickFrequency()) - calibration_start
            cv2.putText(
                frame,
                f"Calibrating... {elapsed:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"Tracked points: {len(new_points)}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"Samples: {len(calibration_features)}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            if elapsed >= CALIBRATION_TIME_SEC:
                if len(calibration_features) >= 20:
                    try:
                        mean_vec, pca, baseline_deviation = calibrate_pca(calibration_features)
                        calibrated = True
                        deviation_buffer = []
                        print(f"Calibration complete. Baseline deviation: {baseline_deviation:.3f}")
                    except ValueError as e:
                        print(f"Calibration failed: {e}")
                        calibration_features = []
                        calibration_start = None
                        prev_points = None
                else:
                    print("Not enough calibration samples collected. Restarting calibration.")
                    calibration_features = []
                    calibration_start = None
                    prev_points = None

            cv2.imshow("Posture Analyzer", frame)
            continue

        if prev_points is None or len(prev_points) < MIN_TRACKED_POINTS:
            prev_points = detect_keypoints(frame)
            prev_frame = frame.copy()

            cv2.putText(
                frame,
                "Monitoring... detecting points",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.imshow("Posture Analyzer", frame)
            continue

        # track points
        new_points, _ = track_keypoints(prev_frame, frame, prev_points)
        prev_frame = frame.copy()

        if len(new_points) < MIN_TRACKED_POINTS:
            prev_points = None
            deviation_buffer.clear()

            cv2.putText(
                frame,
                "Monitoring... re-detecting points",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.imshow("Posture Analyzer", frame)
            continue

        prev_points = new_points.reshape(-1, 1, 2)

        feature_vec = compute_feature_vector(new_points)
        deviation = compute_deviation(feature_vec, mean_vec, pca, baseline_deviation)

        deviation_buffer.append(deviation)
        if len(deviation_buffer) > ROLLING_BUFFER_SIZE:
            deviation_buffer.pop(0)

        smoothed = np.mean(deviation_buffer) if len(deviation_buffer) > 0 else 0.0

        for pt in new_points:
            cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

        status_text = "Good Posture" if smoothed < DEVIATION_THRESHOLD else "Bad Posture"
        color = (0, 255, 0) if status_text == "Good Posture" else (0, 0, 255)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(
            frame,
            f"Deviation: {smoothed:.2f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Tracked points: {len(new_points)}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Posture Analyzer", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
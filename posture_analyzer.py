import cv2
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
MIN_CONTOUR_AREA = 4000
ANGLE_THRESHOLD_DEG = 10
SHOW_MASK = True


def get_center_roi(frame):
    """
    Restrict detection to the central region where the person is expected.
    """
    h, w = frame.shape[:2]
    x1 = int(w * 0.20)
    x2 = int(w * 0.80)
    y1 = int(h * 0.10)
    y2 = int(h * 0.95)
    return x1, y1, x2, y2


def draw_center_roi(frame):
    x1, y1, x2, y2 = get_center_roi(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.putText(frame, "Detection Region", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)


def preprocess_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_foreground_mask(frame, background):
    """
    Compute absolute difference from a stored background frame.
    """
    diff = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold difference image
    _, mask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    mask = preprocess_mask(mask)

    # Keep only the center region
    x1, y1, x2, y2 = get_center_roi(frame)
    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    return mask


def get_largest_contour(mask, min_area=MIN_CONTOUR_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    return largest


def contour_to_points(contour):
    return contour.reshape(-1, 2).astype(np.float32)


def compute_body_axis(contour):
    """
    PCA on contour points to get the body axis.
    Chooses the eigenvector that is most aligned with vertical,
    not just the one with the largest eigenvalue.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return None, None, None

    mean = np.mean(pts, axis=0)
    centered = pts - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Two eigenvectors: choose the one more aligned with vertical
    v1 = eigvecs[:, 0]
    v2 = eigvecs[:, 1]

    vertical = np.array([0.0, 1.0])

    if abs(np.dot(v1, vertical)) > abs(np.dot(v2, vertical)):
        direction = v1
    else:
        direction = v2

    direction = direction / np.linalg.norm(direction)

    # Force line to point upward/downward consistently
    if direction[1] < 0:
        direction = -direction

    cx, cy = mean

    cos_theta = np.clip(abs(np.dot(direction, vertical)), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    return (int(cx), int(cy)), direction, angle_deg


def draw_axis(frame, center, direction, length=220, color=(0, 255, 0), thickness=3):
    cx, cy = center
    dx, dy = direction

    x1 = int(cx - length * dx)
    y1 = int(cy - length * dy)
    x2 = int(cx + length * dx)
    y2 = int(cy + length * dy)

    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    background = None
    calibrated = False
    baseline_angle = None

    print("Instructions:")
    print("1. Start program and leave camera view for 2 seconds.")
    print("2. Press B to capture empty background.")
    print("3. Sit in the detection region.")
    print("4. Press C while sitting straight to calibrate.")
    print("5. Press Q or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to grab frame.")
            break

        display = frame.copy()
        draw_center_roi(display)

        contour = None
        mask = None

        if background is not None:
            mask = get_foreground_mask(frame, background)
            contour = get_largest_contour(mask)

            if contour is not None:
                cv2.drawContours(display, [contour], -1, (255, 200, 0), 2)

                center, direction, angle_deg = compute_body_axis(contour)

                if center is not None:
                    if calibrated and baseline_angle is not None:
                        deviation = abs(angle_deg - baseline_angle)
                        good_posture = deviation < ANGLE_THRESHOLD_DEG
                        status = "Good Posture" if good_posture else "Bad Posture"
                        color = (0, 255, 0) if good_posture else (0, 0, 255)
                    else:
                        deviation = angle_deg
                        status = "Press C to calibrate"
                        color = (0, 255, 255)

                    draw_axis(display, center, direction, length=220, color=color, thickness=3)

                    cv2.putText(display, f"Angle from vertical: {angle_deg:.1f} deg",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if calibrated and baseline_angle is not None:
                        cv2.putText(display, f"Deviation: {deviation:.1f} deg",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display, status,
                                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    else:
                        cv2.putText(display, "Sit straight, then press C",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display, "Body not detected clearly",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Press B to capture empty background",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Step out of the frame first",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(display, "B = background   C = calibrate   Q/ESC = quit",
                    (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2)

        cv2.imshow("Posture Line Analyzer", display)
        if SHOW_MASK and mask is not None:
            cv2.imshow("Foreground Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('b'):
            background = frame.copy()
            calibrated = False
            baseline_angle = None
            print("Background captured.")

        elif key == ord('c'):
            if contour is not None:
                _, _, angle_deg = compute_body_axis(contour)
                if angle_deg is not None:
                    baseline_angle = angle_deg
                    calibrated = True
                    print(f"Calibrated baseline angle: {baseline_angle:.2f} deg")
                else:
                    print("Could not calibrate: axis not found.")
            else:
                print("Could not calibrate: body contour not found.")

        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2
import numpy as np

CARD_WIDTH_CM = 8.56
CARD_HEIGHT_CM = 5.398
PIXELS_PER_CM_DEFAULT = 35.0
MIN_CONTOUR_AREA = 3000
SMOOTHING_FRAMES = 20


class VirtualTapeMeasure:

    def __init__(self):
        # Try different backends and indices for macOS robustness
        self.cap = None
        for index in [0, 1]:
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                self.cap = cap
                break
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.pixels_per_cm = PIXELS_PER_CM_DEFAULT
        self.calibrating = False
        self.history_w = []
        self.history_h = []
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False
        )
        self.warmup_frames = 60
        self.frame_count = 0

    def find_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25, C=5
        )

        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.005)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(thresh, fg_mask)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def filter_contours(self, contours, frame_shape):
        h, w = frame_shape[:2]
        margin = 15
        good = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, cw, ch = cv2.boundingRect(c)
            if x <= margin or y <= margin or (x + cw) >= (w - margin) or (y + ch) >= (h - margin):
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.4:
                continue

            good.append(c)

        return good

    def draw_and_measure(self, frame, contours):
        filtered = self.filter_contours(contours, frame.shape)

        if not filtered:
            self.history_w.clear()
            self.history_h.clear()
            return

        biggest = max(filtered, key=cv2.contourArea)
        rect = cv2.minAreaRect(biggest)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        (width_px, height_px) = rect[1]
        if width_px == 0 or height_px == 0:
            return

        if self.calibrating:
            long_side = max(width_px, height_px)
            short_side = min(width_px, height_px)
            ppcm_w = long_side / CARD_WIDTH_CM
            ppcm_h = short_side / CARD_HEIGHT_CM
            self.pixels_per_cm = (ppcm_w + ppcm_h) / 2.0

            cv2.drawContours(frame, [box], 0, (255, 180, 0), 3)

            text = f"CALIBRATING: {self.pixels_per_cm:.1f} px/cm"
            cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            info = "Place a bank card and press 'C' again to finish."
            cv2.putText(frame, info, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, info, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            return

        self.history_w.append(width_px)
        self.history_h.append(height_px)

        if len(self.history_w) > SMOOTHING_FRAMES:
            self.history_w.pop(0)
            self.history_h.pop(0)

        median_w = float(np.median(self.history_w))
        median_h = float(np.median(self.history_h))

        width_cm = median_w / self.pixels_per_cm
        height_cm = median_h / self.pixels_per_cm

        cv2.drawContours(frame, [box], 0, (0, 255, 0), 3)
        self._draw_dimension_lines(frame, box)

        cx = int(rect[0][0])
        cy = int(rect[0][1])
        label = f"{width_cm:.1f} x {height_cm:.1f} cm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = cx - tw // 2
        ty = cy + th // 2

        cv2.rectangle(frame, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    
    def _draw_dimension_lines(self, frame, box):
        def midpoint(p1, p2):
            return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

        mid_top = midpoint(box[0], box[1])
        mid_right = midpoint(box[1], box[2])
        mid_bottom = midpoint(box[2], box[3])
        mid_left = midpoint(box[3], box[0])

        cv2.line(frame, mid_top, mid_bottom, (255, 0, 0), 1)
        cv2.line(frame, mid_left, mid_right, (255, 0, 0), 1)

        for pt in [mid_top, mid_bottom, mid_left, mid_right]:
            cv2.circle(frame, pt, 4, (255, 0, 0), -1)

    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        info_lines = [
            f"Calibration: {self.pixels_per_cm:.1f} px/cm",
            "ESC=Exit | C=Calibrate | R=Reset",
        ]

        y0 = h - 20
        for i, line in enumerate(reversed(info_lines)):
            y = y0 - i * 25
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if self.frame_count < self.warmup_frames:
            pct = int(self.frame_count / self.warmup_frames * 100)
            cv2.putText(frame, f"Warming up... {pct}%", (w // 2 - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, f"Warming up... {pct}%", (w // 2 - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    def run(self):
        if not self.cap.isOpened():
            print("ERROR: Camera could not be initialized.")
            print("Check if another application is using the camera,")
            print("or grant permission in System Preferences -> Privacy -> Camera.")
            return

        print("=== Virtual Tape Measure started ===")
        print("ESC  - Exit")
        print("C    - Calibrate (place a bank card on the surface)")
        print("R    - Reset calibration")
        print("====================================")

        try:
            # Wait for the first valid frame (camera warmup)
            max_retries = 30
            for _ in range(max_retries):
                success, frame = self.cap.read()
                if success:
                    break
                cv2.waitKey(100)
            
            if not success:
                print("ERROR: Camera is opened but could not read frames.")
                return

            while True:
                success, frame = self.cap.read()
                if not success:
                    break

                self.frame_count += 1
                contours = self.find_objects(frame)

                if self.frame_count > self.warmup_frames:
                    self.draw_and_measure(frame, contours)

                self.draw_hud(frame)
                cv2.imshow('Virtual Tape Measure', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('c') or key == ord('C'):
                    self.calibrating = not self.calibrating
                    if self.calibrating:
                        print("[*] CALIBRATION mode - place a bank card on the surface.")
                    else:
                        print(f"[*] Calibration DONE. New coefficient: {self.pixels_per_cm:.2f} px/cm")
                elif key == ord('r') or key == ord('R'):
                    self.pixels_per_cm = PIXELS_PER_CM_DEFAULT
                    self.history_w.clear()
                    self.history_h.clear()
                    print(f"[*] Calibration reset to {PIXELS_PER_CM_DEFAULT} px/cm")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("=== Virtual Tape Measure stopped ===")


if __name__ == "__main__":
    measure_tool = VirtualTapeMeasure()
    measure_tool.run()
import cv2

class Renderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_match = (0, 255, 0)      # 초록색 (매칭됨)
        self.color_unknown = (0, 0, 255)    # 빨간색 (Unknown)

    def draw_info(self, frame, label, similarity):
        """
        프레임 좌측 상단에 정보를 표시합니다.
        """
        color = self.color_match if label != "Unknown" else self.color_unknown
        text = f"ID: {label} ({similarity:.2f})"
        
        cv2.putText(frame, text, (20, 50), self.font, 1.2, color, 2, cv2.LINE_AA)
        
        return frame

    def show(self, frame, window_name="Cat Re-ID"):
        cv2.imshow(window_name, frame)

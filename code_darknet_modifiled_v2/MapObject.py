class MapObject:
    def __init__(self, class_name, confidence, left, top, right, bottom):
        self.class_name = class_name
        self.confidence = confidence
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

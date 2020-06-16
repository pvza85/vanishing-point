import numpy as np


class Intersection:
    """
    A class of intersection including 2 lines that their intersection point (x, y).
    Also I calculate lines length and intersection angle.
    """
    def __init__(self, line1, line2, shape, min_angle=0, max_angle=22.5, grid_size=100):
        self.line1 = line1
        self.line2 = line2
        self.shape = shape
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.x, self.y = self.line_intersection()
        self.fit = True
        self.line1_len = self.get_len(line1)
        self.line2_len = self.get_len(line2)
        self.total_len = self.line1_len + self.line2_len
        self.grid_size = grid_size
        self.grid = [0, 0]
        self.angle = self.get_angle(line1, line2)
        # temporary:
        self.x_org = self.x
        self.y_org = self.y
        self.line1_org = self.line1
        self.line2_org = self.line2

    def print_shifts(self):
        print(f"x: {self.x_org} -> {self.x}")
        print(f"y: {self.y_org} -> {self.y}")
        print(f"line1: {self.line1_org} -> {self.line1}")
        print(f"line2: {self.line2_org} -> {self.line2}")

    def line_intersection(self):
        line1 = self.line1
        line2 = self.line2
        x = [line2[0] - line1[0], line2[1] - line1[1]]
        d1 = [line1[2] - line1[0], line1[3] - line1[1]]
        d2 = [line2[2] - line2[0], line2[3] - line2[1]]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if cross == 0:
            return None
        t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
        return line1[0] + d1[0] * t1, line1[1] + d1[1] * t1

    @staticmethod
    def get_len(line):
        return np.sqrt(np.power((line[2] - line[0]), 2) + np.power((line[3] - line[1]), 2))

    @staticmethod
    def get_angle(l1, l2):
        def get_vec(line):
            return [line[2] - line[0], line[3] - line[1]]
        def get_len(line):
            return np.sqrt(np.power((line[2] - line[0]), 2) + np.power((line[3] - line[1]), 2))
        cos_theta = np.dot(get_vec(l1), get_vec(l2)) / (get_len(l1) * get_len(l2))
        angle = np.degrees(np.arccos(cos_theta))
        if angle > 90:
            angle = 180-90
        return angle

    def set_grid(self, pad_ratio):
        shape = ((1+pad_ratio) * self.shape[0], (1+pad_ratio) * self.shape[1])
        grid_x = int((self.x + 1) / (shape[0] / np.sqrt(self.grid_size)))
        grid_y = int((self.y + 1) / (shape[1] / np.sqrt(self.grid_size)))
        self.grid = [grid_x, grid_y]

    def shift(self, pad_ratio):
        def shift_point(a, b):
            return [int(np.round(a + pad_ratio * self.shape[1])), int(np.round(b + pad_ratio * self.shape[0]))]

        [self.x, self.y] = shift_point(self.x, self.y)
        self.line1 = shift_point(self.line1[0], self.line1[1]) + shift_point(self.line1[2], self.line1[3])
        self.line2 = shift_point(self.line2[0], self.line2[1]) + shift_point(self.line2[2], self.line2[3])
        self.set_grid(pad_ratio)
        return 0 < self.x < self.shape[0] + pad_ratio * self.shape[0] \
               and 0 < self.y < self.shape[1] + pad_ratio * self.shape[1] \
               and self.min_angle < self.angle < self.max_angle

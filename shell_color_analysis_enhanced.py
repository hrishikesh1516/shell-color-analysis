import cv2
import numpy as np
from collections import Counter
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

class ShellColorAnalysis:
    def __init__(self, perceptual_threshold=30):
        self.perceptual_threshold = perceptual_threshold

    def normalize_image(self, image_path):
        # Load image
        image = cv2.imread(image_path)
        # Resize to 1000x1000
        normalized_image = cv2.resize(image, (1000, 1000))
        return normalized_image

    def improve_color_naming(self, color):
        # Improved naming with HSL fallback
        h, l, s = self.rgb_to_hsl(color)
        return self.get_color_name_from_hsl(h, s, l)

    def rgb_to_hsl(self, rgb):
        # Convert RGB to HSL
        r, g, b = rgb / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        h = l = s = (mx + mn) / 2.0

        if mx == mn:
            h = s = 0  # achromatic
        else:
            d = mx - mn
            s = 0 if l == 0 else d / (1 - abs(2 * l - 1))

            if mx == r:
                h = (g - b) / d + (6 if g < b else 0)
            elif mx == g:
                h = (b - r) / d + 2
            elif mx == b:
                h = (r - g) / d + 4
            h /= 6

        return h * 360, s * 100, l * 100  # Scale to [0, 360], [0, 100], [0, 100]

    def get_color_name_from_hsl(self, h, s, l):
        # Dummy implementation of color naming based on HSL
        if s < 10 and l > 90:
            return "White/Cream"
        # More names can be added based on HSL values
        return "Other Color"

    def consolidate_shades(self, colors):
        # Consolidate colors based on perceptual distance
        consolidated = []
        for color in colors:
            if not any(self.color_distance(color, c) < self.perceptual_threshold for c in consolidated):
                consolidated.append(color)
        return consolidated

    def color_distance(self, c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    def analyze_color_spread(self, images):
        color_data = []
        for img_path in images:
            normalized_img = self.normalize_image(img_path)
            avg_color = np.mean(normalized_img.reshape(-1, normalized_img.shape[-1]), axis=0)
            color_data.append(avg_color)
        # Create heatmap and spread index
        self.create_heatmap(color_data)

    def create_heatmap(self, color_data):
        # Implementation for creating heatmap
        plt.hist2d(*zip(*color_data), bins=50, cmap='hot')
        plt.colorbar()
        plt.show()

    def hierarchical_clustering(self, colors):
        # Perform hierarchical clustering
        Z = sch.linkage(colors, 'ward')
        return sch.dendrogram(Z)

    def run_analysis(self, image_paths):
        normalized_colors = []
        for path in image_paths:
            image = self.normalize_image(path)
            avg_color = tuple(np.mean(image.reshape(-1, image.shape[-1]), axis=0))
            normalized_colors.append(avg_color)

        consolidated_colors = self.consolidate_shades(normalized_colors)
        self.analyze_color_spread(image_paths)
        return consolidated_colors

# Usage
if __name__ == '__main__':
    analyzer = ShellColorAnalysis(perceptual_threshold=30)
    image_files = ['image1.jpg', 'image2.jpg']  # Add paths to your images
    results = analyzer.run_analysis(image_files)
    print("Consolidated Colors:", results)
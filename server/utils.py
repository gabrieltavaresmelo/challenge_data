import numpy as np

colorBlue=(255, 0, 0)
colorRed=(0, 0, 255)
colorGreen=(0, 255, 0)

def load_data(base_path, test_path):
    base_data = np.load(base_path)
    test_data = np.load(test_path)

    base_images = base_data['images']
    test_images = test_data['images']
    base_gps = base_data['gps']
    test_gps = test_data['gps']
    base_compass = base_data['compass']
    test_compass = test_data['compass']
    
    return base_images, test_images, base_gps, test_gps, base_compass, test_compass

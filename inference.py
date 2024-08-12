import cv2
import numpy as np
import torch
from lib.networks import EMCADNet
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


model = EMCADNet(num_classes = 2)
model.load_state_dict(torch.load("data/huaxiproj/EMCAD/ca/best_model.pth"))

output_size = 256

image_paths = ["tmp.png"]
model.eval()
with torch.no_grad():
    for im_p in image_paths:
        image = cv2.imread(im_p, 0)
        x, y = image.shape
        image = zoom(image, (output_size / x, output_size / y), order=3)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        image = image.unsqueeze(0)
        masks = model(image)
        mask = torch.argmax(torch.softmax(masks[-1], dim=1), dim=1, keepdim=True).squeeze()
        plt.imsave("tmp_mask.png", mask.numpy(), cmap="gray")
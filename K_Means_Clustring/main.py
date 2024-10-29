import numpy as np
from PIL import Image

# All pixels cropped image
allPixels = []

for picture in range(5):

    # Open images
    image_path = "D://University//Term 7//Linear Algebra//Project//k-means-clustring//DataSet//usps_" + str(picture + 1) + ".jpg"
    image = Image.open(image_path)

    for x in range(33):
        for y in range(34):

            if(x == 32 and y == 12):
                break;

            crop_area = (x * 16, y * 16,(x + 1) * 16,(y + 1) * 16)
            cropped_image = image.crop(crop_area)

            # cropped_image.save("D://University//Term 7//Linear Algebra//Project//k-means-clustring//DataSet//Cropped_Image//usps_cropped_" + str(picture + 1) + "_" + str(x) + "_" + str(y) + ".jpg")

            pixel = np.array(cropped_image)

            shape = np.reshape(pixel, (256, 1))

            allPixels.append(shape)
import cv2
import argparse 

def arg_parse():
    """
        Parsing arguments to resizing
    """
    parser = argparse.ArgumentParser(description='Resizing inputs for the detector')

    parser.add_argument("--image", dest = "image", help = "Image for the resizing opn",
                        default = "img.png", type= str)

    return parser.parse_args()

args = arg_parse()

img = cv2.imread(args.image)
resized = cv2.resize(img, (327, 324), interpolation = cv2.INTER_AREA)
print("Resized dim: ",resized.shape)

filename = "resize_" + args.image
cv2.imwrite(filename, resized)
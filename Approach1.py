import sys                                          # System bindings
import cv2                                          # OpenCV bindings
import numpy as np

#https://blog.danielparker.com.au/background-colour-detection-using-opencv-and-python-22ed8655b243
 
class ColorAnalyser():
    def __init__(self, imageLoc):
        self.src = cv2.imread(imageLoc, 1)          # Reads in image source
        self.colors_count = {}                      # Empty dictionary container to hold the colour frequencies
 
    def count_colors(self):
        (channel_b, channel_g, channel_r) = cv2.split(self.src) # Splits image Mat into 3 color channels in individual 2D arrays
 
        channel_b = channel_b.flatten()                         # Flattens the 2D single channel array so as to make it easier to iterate over it
        channel_g = channel_g.flatten()                         #                   ""
        channel_r = channel_r.flatten()                         #                   ""
 
        for i in xrange(len(channel_b)):
            RGB = "(" + str(channel_r[i]) + "," + str(channel_g[i]) + "," + str(channel_b[i]) + ")"
            if RGB in self.colors_count:
                self.colors_count[RGB] += 1
            else:
                self.colors_count[RGB] = 1
 
        print "Colours counted"
 
    def show_colors(self):
        for keys in sorted(self.colors_count, key=self.colors_count.__getitem__):       # Sorts dictionary by value
            print keys, ": ",self.colors_count[keys]                  # Prints 'key: value'
 
    def main(self):
        if (self.src.all() == None):                                  # Checks if an image was actually loaded and errors if it wasn't
            print "No image data. Check image location for typos"
        else:
            self.count_colors()                             # Counts the amount of instances of RGB values within the image
            self.show_colors()                              # Sorts and shows the colors ordered from least to most often occurance
            cv2.waitKey(0)                                  # Waits for keypress before closing
 
if __name__ == "__main__":
    if (len(sys.argv) != 2):                                    # Checks if image was given as cli argument
        print "error: syntax is 'python main.py /example/image/location.jpg'"
    else:
        Analyser = ColorAnalyser(sys.argv[1])
        Analyser.main()
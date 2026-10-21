# barnacle_counter: the small package the notebook and scripts import.
# frames  -> straighten the green frame
# labels  -> turn the mask outlines into one point per barnacle
# tiles   -> cut the image into overlapping tiles, run the model, stitch dots back together
# detector-> load a model (NumPy or Ultralytics) and count a photo
# metrics -> match dots to barnacles, F1, count error
# audit   -> random-tile check that turns a model count into count +/- interval
# marks   -> read hand marks (red lines) and map a crop back into its full photo
# plots   -> overlays
from .frames import read_rgb, rectify
from .labels import load_reference, in_box
from .tiles import windows, tile_predict
from .detector import Detector, count_photo
from .metrics import match_objects, metrics, count_error
from .audit import tile_counts, corrected_count

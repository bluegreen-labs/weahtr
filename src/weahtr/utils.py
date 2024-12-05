import numpy as np
import cv2

def binarize(
  im: np.ndarray
    ) -> np.ndarray:
    
    if not isinstance(im, np.ndarray) or im.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale NumPy array.")
    
    # Adaptive thresholding
    im = cv2.adaptiveThreshold(
        im,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Median blurring
    im = cv2.medianBlur(im, 3)
    
    if verbose:
        print("Applied adaptive thresholding and median blurring.")
    
    return im

def load_guides(filename, mask_name):
   # check if the guides file can be read
   # if not return error
   try:
    guides = []
    file = open(u''+filename,'r')
    lines = file.readlines()
    for line in lines:
     if line.find("Guide:" + mask_name) > -1:
       data = line.split('|')
       data[0] = data[0].split(":")
       data[1] = data[1].split(",")
       data[2] = data[2].split(",")
       data[3] = data[3].split(",")
       guides.append(data)
    file.close()
    return guides
   except:
    print("No subset location file found!")
    print("looking for: " + mask_name + ".csv")
    exit()

def print_labels(im, locations, df):
  
  # split out the locations
  # convert to integer
  x = np.asarray(locations[0][3], dtype=float)
  x = x.astype(int)
  x = np.sort(x)
  y = np.asarray(locations[0][2], dtype=float)
  y = y.astype(int)
  y = np.sort(y)
  
  # loop over all rows
  for i, row in df.iterrows():
   y_value = int(row['row'])
   x_value = int(row['col'])
   label = row['text']
   conf = row['conf']
   
   center_x = x[x_value -1]
   center_y = y[y_value]
   
   try:
     if conf > 85:
      cv2.putText(im, str(label) ,(center_x, center_y),
        cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),9,cv2.LINE_AA)
     else:
      cv2.putText(im, str(label) ,(center_x, center_y),
        cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),9,cv2.LINE_AA) 
   except:
    continue
  return im

# TrOCR model training and inference class
# model training routines depend on the
# underlying model and framework and are
# described in other files
class model:
  
  def __init__(self, model, config, images, labels=None, **kwargs):
    self.model = model
    self.image_path = images
    self.labels = labels

  def train(self, transform = None):
    if self.model == "trorc":
      train_trocr(self.image_path, self.labels)
    else:
      print("bla")
    
  def predict(self, image):
    print("predicting")

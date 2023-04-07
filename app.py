from fastai.vision.all import *
from fastai.vision.utils import *
import matplotlib.pyplot as plt

learn = load_learner('components.pkl')
categories = ('keyboard', 'monitor', 'mouse')

def classify_images(img):
    pred_class,pred_idx,probs = learn.predict(img)
    # print('its a', pred_class)
    # print(pred_idx)
    # print(probs)
    return dict(zip(categories, map(float, probs)))


# Mouse Predictions
imgMouse = plt.imread('mouse.jpg')
mousePredictions = classify_images(imgMouse)

# Keyboard Predictions
imgKeyboard = plt.imread('keyboard.jpg')
keyboardPredictions = classify_images(imgKeyboard)

# # Monitor Predictions
imgMonitor = plt.imread('monitor.jpg')
monitorPredictions = classify_images(imgMonitor)

print('Mouse Predictions: ', mousePredictions)
print('Keyboard Predictions: ', keyboardPredictions)
print('Monitor Predictions: ', monitorPredictions)


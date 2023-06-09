import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets , svm , metrics , tree , naive_bayes
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

ds = datasets.load_digits()

_,axes = plt.subplots(nrows=1,ncols=4,figsize=(10,4))
for axis,image,label in zip(axes,ds.images,ds.target):
  axis.set_axis_off()
  axis.imshow(image,cmap="Blues_r")
  axis.set_title("Training : {}".format(label))

n_samples = len(ds.images)
data = ds.images.reshape(n_samples,-1)
model = svm.SVC(kernel="linear")
X_train,X_test,Y_train,Y_test = train_test_split(data,ds.target,test_size=0.2,shuffle=True)
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

_,axes = plt.subplots(nrows=1,ncols=4,figsize=(10,4))
for axis,image,label in zip(axes,X_test,y_pred):
  axis.set_axis_off()
  image = image.reshape(8,8)
  axis.imshow(image,cmap="gray")
  axis.set_title("Prediction : {}".format(label))

print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(Y_test, y_pred)}\n"
)

print("\n")
disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, y_pred)
disp.figure_.suptitle("Confusion Matrix")

plt.show()
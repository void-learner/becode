import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load handwritten digits dataset
digits = datasets.load_digits()

# Flatten the images 
X = digits.data
y = digits.target

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='rbf', gamma=0.05, C=10)
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Function to visualize and predict single digits 
def predict_digit_image(index):
    image = digits.images[index]
    data = digits.data[index].reshape(1, -1)
    data = scaler.transform(data)
    prediction = svm_model.predict(data)[0]

    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted Digit: {prediction}")
    plt.axis('off')
    plt.show()

# Example usage:
n = int(input("Chosse no. from 0-1796: "))
predict_digit_image(n)  

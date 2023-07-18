from AdalineSGD import AdalineSGD
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tkinter import messagebox, ttk
from tkinter import *

if __name__ == '__main__':
    dataset = load_breast_cancer()
    # print dataset's about
    print(f"Dataset's about: {dataset.DESCR}")
    # make y from 0 and 1 to -1 and 1
    y = np.where(dataset.target == 0, -1, 1)
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target

    print(f'\nThe last 2 predictions are: {y[-2:]}')

    # save the last 2 rows of dataset.data to csv file with headers (Relevant features)
    np.savetxt("breast_cancer.csv", dataset.data[-2:, :], delimiter=",", header="mean radius,mean texture,mean perimeter,mean area,mean smoothness,mean compactness,mean concavity,mean concave points,mean symmetry,mean fractal dimension,radius error,texture error,perimeter error,area error,smoothness error,compactness error,concavity error,concave points error,symmetry error,fractal dimension error,worst radius,worst texture,worst perimeter,worst area,worst smoothness,worst compactness,worst concavity,worst concave points,worst symmetry,worst fractal dimension")

    # take all except the last 2 rows
    df = df[:-2]
    y = y[:-2]

    print(f"Feature name: {dataset.feature_names}")
    print(f"Data shape: {dataset.data.shape}")

    # Choose X by correlation with the target variable
    corr_matrix = df.corr()
    corr_target = abs(corr_matrix['target'])
    relevant_features = corr_target[corr_target > 0.5].index.tolist()
    relevant_features.remove('target')
    X = df[relevant_features].values

    # Standardize all columns of X
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    print(f"Data shape: {X.shape}")
    print(f"Relevant features: {relevant_features}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.05, random_state=0)
    print(y_test.shape, y_train.shape)
    
    # Create a form to enter the parameters of AdalineSGD and Relevant features
    root = Tk()
    root.title('Predict Breast Cancer using AdalineSGD')
    root.geometry("800x500")
    root.configure(background='cyan')

    # Create a label for eta with font size 16
    eta_label = Label(root, text="eta", bg="cyan", fg="black", font=("Helvetica", 16))
    eta_label.grid(row=0, column=0, sticky=W)

    # Create a label for n_iter
    n_iter_label = Label(root, text="n_iter", bg="cyan", fg="black", font=("Helvetica", 16))
    n_iter_label.grid(row=1, column=0, sticky=W)
    # Create a label for accuracy
    accuracy_label = Label(root, text="accuracy", bg="cyan", fg="black", font=("Helvetica", 16))
    accuracy_label.grid(row=3, column=0, sticky=W)

    # Create a text box for eta
    eta_text = Text(root, height=1, width=10, font=("Helvetica", 16))
    eta_text.grid(row=0, column=1, sticky=W)
    # Create a text box for n_iter
    n_iter_text = Text(root, height=1, width=10, font=("Helvetica", 16))
    n_iter_text.grid(row=1, column=1, sticky=W)
    # Create text box for accuracy (read only)
    accuracy_text = Text(root, height=1, width=10, font=("Helvetica", 16))
    accuracy_text.grid(row=3, column=1, sticky=W)

    # auto create labels and text boxes for relevant features
    for i in range(len(relevant_features)):
        label = Label(root, text=relevant_features[i], bg="cyan", fg="black", font=("Helvetica", 16))
        label.grid(row=i, column=2, sticky=W)
        text = Text(root, height=1, width=10, font=("Helvetica", 16))
        text.grid(row=i, column=3, sticky=W)
    
    # Create a button to train the model
    def train_model():
        # get n_iter and eta from text boxes
        #Text.get() missing 1 required positional argument: 'index1'
        n_iter_value = int(n_iter_text.get(1.0, END))
        eta_value = float(eta_text.get(1.0, END))
        adaline = AdalineSGD(n_iter=n_iter_value, eta=eta_value, random_state=1).fit(X_train, y_train)
        # save model to file
        with open('model.pkl', 'wb') as f:
            pickle.dump(adaline, f)
    train_button = Button(root, text="Train", font=("Helvetica", 16), command=lambda: train_model())
    train_button.grid(row=0, column=4, sticky=W)

    # Create a button to test the model
    def test_model():
        with open('model.pkl', 'rb') as f:
            adaline = pickle.load(f)
        y_pred = adaline.predict(X_test)
        accuracy_text.delete(1.0, END)
        accuracy_text.insert(END, f"{np.sum(y_pred == y_test) / len(y_test)}")

    test_button = Button(root, text="Test", font=("Helvetica", 16), command=lambda: test_model())
    test_button.grid(row=1, column=4, sticky=W)

    # Create a button to plot cost function
    def plot_cost_function():
        # Load model from file
        with open('model.pkl', 'rb') as f:
            _adaline = pickle.load(f)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
        ax.plot(range(1, len(_adaline.cost_) + 1), _adaline.cost_, marker='o')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Sum-squared-error')
        ax.set_title(f'Adaline - Learning rate {float(eta_text.get(1.0, END))}')
        plt.show()
    plot_button = Button(root, text="Plot cost function", font=("Helvetica", 16), command=lambda: plot_cost_function())
    plot_button.grid(row=2, column=4, sticky=W)

    # Create a button to predict breat cancer from text boxes
    def predict_breast_cancer():
        # get values from text boxes
        values = []
        for i in range(len(relevant_features)):
            values.append(float(root.grid_slaves(i, 3)[0].get(1.0, END)))
        # standardize values
        values = scaler.transform([values])
        # load model from file
        with open('model.pkl', 'rb') as f:
            adaline = pickle.load(f)
        # predict
        y_pred = adaline.predict([values])
        # show result to message box
        if y_pred == 1:
            messagebox.showinfo("Prediction", f"Prediction: {y_pred} (malignant)")
        else:
            messagebox.showinfo("Prediction", f"Prediction: {y_pred} (benign)")
    predict_button = Button(root, text="Predict", font=("Helvetica", 16), command=lambda: predict_breast_cancer())
    predict_button.grid(row=3, column=4, sticky=W)
    root.mainloop()
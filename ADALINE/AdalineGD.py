import numpy as np


class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
    
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    radom_state : int
        Random number generator seed for random weight initialization.
    
    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.
    
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ------------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y: array-like, shape = [n_samples]
            Target values.
        
        Returns
        ------------
        self: object
        
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import pandas as pd
    import pickle
    from tkinter import ttk
    from tkinter import *
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    dataset = load_breast_cancer()
    # make y from 0 and 1 to -1 and 1
    y = np.where(dataset.target == 0, -1, 1)
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
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
    window = Tk()
    window.title("Adaline using Gradient Descent")
    window.geometry('500x300')  # width x height

    # create labels
    ttk.Label(window, text="Number of iterations").grid(column=0, row=0, padx=10, pady=10)
    ttk.Label(window, text="Learning rate").grid(column=0, row=1, padx=10, pady=10)
    ttk.Label(window, text="Accuracy").grid(column=0, row=3, padx=10, pady=10)

    # create textboxes
    n_iter = ttk.Entry(window, width=10)
    n_iter.grid(column=1, row=0, padx=10, pady=10)

    eta = ttk.Entry(window, width=10)
    eta.grid(column=1, row=1, padx=10, pady=10)

    accuracy = ttk.Entry(window, width=10, state='readonly')
    accuracy.grid(column=1, row=3, padx=10, pady=10)

    def train_model():
        n_iter_value = int(n_iter.get())
        eta_value = float(eta.get())
        adaline = AdalineGD(eta_value, n_iter_value).fit(X_train, y_train)
        # save model to file
        with open('model.pkl', 'wb') as f:
            pickle.dump(adaline, f)

    ttk.Button(window, text="Train", command=train_model).grid(column=0, row=2, padx=10, pady=10)

    def plot_cost_function():
        # Load model from file
        with open('model.pkl', 'rb') as f:
            _adaline = pickle.load(f)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
        ax.plot(range(1, len(_adaline.cost_) + 1), _adaline.cost_, marker='o')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Sum-squared-error')
        ax.set_title(f'Adaline - Learning rate {float(eta.get())}')
        plt.show()

    ttk.Button(window, text="Plot cost function", command=plot_cost_function).grid(column=1, row=2, padx=10, pady=10)

    def test():
        with open('model.pkl', 'rb') as f:
            _adaline = pickle.load(f)
        y_pred = _adaline.predict(X_test)
        # calculate accuracy using mean
        accuracy_value = np.mean(y_pred == y_test)
        # set accuracy_value to accuracy textbox
        accuracy.configure(state='normal')
        accuracy.delete(0, END)
        accuracy.insert(0, f'{accuracy_value:.2%}')
        accuracy.configure(state='readonly')

    ttk.Button(window, text="Test", command=test).grid(column=2, row=2, padx=10, pady=10)

    # show the window
    window.mainloop()

    # ----------------- Test (for iris dataset)----------------- #
    """
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    print(X)
    print(y)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    # plt.savefig('images/02_11.png', dpi=300)
    plt.show()

    # Improving gradient descent through feature scaling
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    ada_gd = AdalineGD(n_iter=15, eta=0.01)
    ada_gd.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada_gd)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.xlabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()

    # save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(ada1, f)
    # create form using tkinter for input new record (4 cols) for classification
    root = Tk()
    root.title("Iris Classification")
    root.geometry("400x400")
    root.configure(background="light blue")
    # create a label for each entry
    Label(root, text="Sepal Length", bg="light blue").grid(row=0, sticky=W)
    Label(root, text="Sepal Width", bg="light blue").grid(row=1, sticky=W)
    Label(root, text="Petal Length", bg="light blue").grid(row=2, sticky=W)
    Label(root, text="Petal Width", bg="light blue").grid(row=3, sticky=W)
    # create a text entry box for each label
    sepal_length = Entry(root)
    sepal_length.grid(row=0, column=1)
    sepal_width = Entry(root)
    sepal_width.grid(row=1, column=1)
    petal_length = Entry(root)
    petal_length.grid(row=2, column=1)
    petal_width = Entry(root)
    petal_width.grid(row=3, column=1)
    # create a submit button
    def submit():
        # load the model from disk
        loaded_model = pickle.load(open('model.pkl', 'rb'))
        result = loaded_model.predict(np.array([[float(sepal_length.get()), float(petal_length.get())]]))
        # result = loaded_model.predict(np.array([[float(sepal_length.get()), float(sepal_width.get()), float(petal_length.get()), float(petal_width.get())]]))
        if result == 1:
            Label(root, text="Iris Setosa", bg="light blue").grid(row=5, column=1, sticky=W)
        else:
            Label(root, text="Iris Versicolour", bg="light blue").grid(row=5, column=1, sticky=W)  
    Button(root, text='Submit', width=20, command=submit).grid(row=4, column=1, sticky=W)
    root.mainloop()
    """
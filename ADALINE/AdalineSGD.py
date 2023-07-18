import numpy as np


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
    
    Parameters
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset.
    shuffle: bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state: int
        Random number generator seed for random weight initialization.
    
    Attributes
    ------------
    w_: 1d-array
        Weights after fitting.
    cost_: list
        Sum-of-squares cost function value averaged over all training samples in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
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
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

if __name__ == '__main__':
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
    window.title("Adaline using Stochastic Gradient Descent")
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
        adaline = AdalineSGD(n_iter=n_iter_value, eta=eta_value, random_state=1).fit(X_train, y_train)
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
        # Load model from file
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
    print(df)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    # Improving gradient descent through feature scaling
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada_sgd)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.tight_layout()  
    plt.show()
    """
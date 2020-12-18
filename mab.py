import numpy as np

class Egreedy:
    def __init__(self, epsilon, arms_number):
        """
        Epsilon Greedy Bandit

        Parameters
        ----------
        epsilon : Epsilon value for the algorithm

        arms_number : Number of arms
        """
        self.e = epsilon
        self.avg_reward = np.zeros(arms_number) #average reward of each arm
        self.n = np.zeros(arms_number)  # number of times each arm was chosen
        self.name = "E-Greedy e=" + str(self.e)


    def select_arm(self, user_features, article_idx, t, features):
        """
        Parameters
        ----------
        user_features : Array of user features

        article_idx : Array of available article indexes to display

        t : round number
   
        features : Array of article features

        Returns
        ------
        index : return index of the selected article
        """
        if np.random.rand() > self.e:
            return np.argmax(self.avg_reward[article_idx])
        else:
            return np.random.randint(low=0, high=len(article_idx))

    def update(self, displayed_idx, reward, user_features, article_idx, features):
        """
        Parameters
        ----------
        displayed_idx : Index of the displayed article

        reward : Binary value if user clicked

        user_features : Array of user features

        articles_idx : Array of article indexes

        features : Array of article features
        """
        i = article_idx[displayed_idx]
        self.n[i] += 1
        self.avg_reward[i] += (reward - self.avg_reward[i]) / self.n[i]

class ThompsonSampling:
    def __init__(self, arms_number):
        self.alpha = np.ones(arms_number)
        self.beta = np.ones(arms_number)
        self.name = "ThompsonSampling"

    def select_arm(self, user_features, article_idx, t, features):
        """
        Parameters
        ----------
        user_features : Array of user features

        article_idx : Array of available article indexes to display

        t : round number

        features : Array of article features

        Returns
        ------
        index : return index of the selected article
        """
        theta = np.random.beta(self.alpha[article_idx], self.beta[article_idx])
        return np.argmax(theta)

    def update(self, displayed_idx, reward, user_features, article_idx, features):
        """
        Parameters
        ----------
        displayed_idx : Index of the displayed article

        reward : Binary value if user clicked

        user_features : Array of user features

        articles_idx : Array of article indexes
        
        features : Array of article features
        """
        i = article_idx[displayed_idx]

        self.alpha[i] += reward
        self.beta[i] += 1 - reward       

class Ucb1:
    def __init__(self, alpha, arms_number):
        self.alpha = alpha
        self.avg_reward = np.zeros(arms_number)  # average reward for each arm
        self.n = np.ones(arms_number)  # number of times each arm was chosen
        self.name = "UCB1 α=" + str(self.alpha)

    def select_arm(self, user_features, article_idx, t, features):
        """
        Parameters
        ----------
        user_features : Array of user features

        article_idx : Array of available article indexes to display

        t : round number

        features : Array of article features

        Returns
        ------
        index : return index of the selected article
        """
        ucb = self.avg_reward[article_idx] + np.sqrt(self.alpha * np.log(t + 1) / self.n[article_idx])
        return np.argmax(ucb)

    def update(self, displayed_idx, reward, user_features, article_idx, features):
        """
        Parameters
        ----------
        displayed_idx : Index of the displayed article

        reward : Binary value if user clicked

        user_features : Array of user features

        articles_idx : Array of article indexes
        
        features : Array of article features
        """
        i = article_idx[displayed_idx]

        self.n[i] += 1
        self.avg_reward[i] += (reward - self.avg_reward[i]) / self.n[i]

class LinUcb:
    def __init__(self, alpha, arms_number):
        d = 12  # total number of features
        self.A = np.array([np.identity(d)] * arms_number)
        self.b = np.zeros((arms_number, d, 1))
        self.alpha = alpha
        self.name = "LinUcb alpha=" + str(self.alpha)


    def select_arm(self, user_features, article_idx, t, features):
        """
        Parameters
        ----------
        user_features : Array of user features

        article_idx : Array of available article indexes to display

        t : round number

        features : Array of article features
        
        Returns
        ------
        index : return index of the selected article
        """
        A = self.A[article_idx]  # (23, 12, 6)
        b = self.b[article_idx]  # (23, 12, 1)
        user = np.array([user_features] * len(article_idx))  # (23, 6)

        A = np.linalg.inv(A)
        x = np.hstack((user, features[article_idx]))  # (23, 12) The vector x summarizes information of both the user u and arm a

        x = x.reshape((len(article_idx), 12, 1))  # (23, 12, 1)

        theta = A @ b  # (23, 12, 1)

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(np.transpose(x, (0, 2, 1)) @ A @ x)
        return np.argmax(p)


    def update(self, displayed_idx, reward, user_features, article_idx, features):
        """
        Parameters
        ----------
        displayed_idx : Index of the displayed article

        reward : Binary value if user clicked

        user_features : Array of user features

        articles_idx : Array of article indexes
        
        features : Array of article features
        """
        a = article_idx[displayed_idx]  # displayed article's index

        x = np.hstack((user_features, features[a]))
        x = x.reshape((12, 1))

        self.A[a] = self.A[a] + x @ np.transpose(x)
        self.b[a] += reward * x

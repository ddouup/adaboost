import numpy as np
import sys

class WeakLearner():
    def __init__(self):
        return

    def fit(self, X, y, weight):
        num = X.shape[0]             # number of instances
        feature_num = X.shape[1]     # number of features

        step_num = 10
        min_error = np.inf
        for i in range(feature_num):
            range_min = X[:,i].min(); range_max = X[:,i].max()
            step_size = (range_max - range_min)/step_num

            self.feature_col = i
            for j in range(step_num+1):
                self.threshold = range_min + j*step_size
                pred = self.predict(X).reshape(-1,1)

                error = np.ones((num,1))
                error[pred == y] = 0

                weight_error = np.sum(weight * error)
                
                '''
                print("feature_col:", i)
                print("threshold:", self.threshold)
                print("weight_error:", weight_error)
                print()
                '''

                if weight_error < min_error:
                    #print("Update feature_col & threshold")
                    #print()
                    min_error = weight_error

                    pred_class = pred.copy()
                    feature_col = self.feature_col
                    threshold = self.threshold

        self.feature_col = feature_col
        self.threshold = threshold

        return self, min_error, pred_class

    def predict(self, X):
        #print("Prediction data size: ", X.shape)
        y_pre = np.array([], dtype=int)
        for i in range(X.shape[0]):
            row = X[i]
            if row[self.feature_col] > self.threshold:
                y_pre = np.append(y_pre, 1)
            else:
                y_pre = np.append(y_pre, -1)

        return y_pre

    def setAlpha(self, a):
        self.alpha = a

    def getAlpha(self):
        return self.alpha

        
class Adaboost():
    def __init__(self, num=10):
        self.itr_num = num

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.labels = np.unique(y_train)
        self.label_num = len(self.labels)       # number of unique labels
        self.num = X_train.shape[0]             # number of instances
        self.feature_num = X_train.shape[1]     # number of features

        self.w = np.full((self.num, 1), 1/self.num) #weight of each data instance
        self.weakLearners = []

        print("Class number: ", self.label_num)
        print("Training data size: ", self.X.shape)
        print()

        for i in range(self.itr_num):
            print("The",i+1,"weak learner")
            learner, error, pred_class= WeakLearner().fit(self.X, self.y, self.w)

            alpha = 1/2 * np.log((1-error)/error)
            print("Alpha:", alpha)
            learner.setAlpha(alpha)
            self.weakLearners.append(learner)

            temp = self.w * np.exp(-alpha*self.y*pred_class)
            self.w = temp/np.sum(temp)
            print("weight:")
            print(self.w)
            print()

        return self   


    def predict(self, X_test):
        print("Test data size: ", X_test.shape)
        y_pre = np.zeros((X_test.shape[0],1))
        for i in range(len(self.weakLearners)):
            pred = self.weakLearners[i].predict(X_test).reshape(-1,1)
            y_pre += self.weakLearners[i].getAlpha()*pred

        y_pre = np.sign(y_pre)

        return y_pre
        

def main():
    #Brown/White = 0, Yellow = 1
    X_train = np.array([
        [3,13,0,0],
        [1,0.01,1,0],
        [1,92.50,2,1],
        [2,33.33,1,0],
        [3,8.99,3,1],
        [2,8.99,0,0],
        [3,13.65,0,0],
        [1,0.01,2,1],
        [1,92.50,2,0],
        [2,33.33,2,1],
        [3,8.99,1,0],
        [3,12.49,2,0]])
    #yes = 1, no = -1
    y_train = np.array([
        [1],
        [-1],
        [-1],
        [-1],
        [-1],
        [1],
        [1],
        [-1],
        [-1],
        [-1],
        [1],
        [-1]])

    #print(X_train)
    #print(y_train)

    model = Adaboost(10).fit(X_train, y_train)
    X_test = X_train = np.array([[3,0,0,0]])
    
    pre = model.predict(X_test)
    print("Prediction result:")
    print(pre)

if __name__ == '__main__':
    main()
# EXERCISE 1
# Perform exploratory data analysis to analyze the dataset. 
# Provide useful visualizations and conclusions from your findings.
import pandas as pd
import numpy as  np 
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# Load HeartFailure.csv from the file directory.
data = pd.read_csv("/home/nikolakis/Desktop/wings_ict_project/HeartFailure.csv")

def printing_our_dataset(data):
    print("\n Exercise 1 :")
    #Print the first 5 rows of the dataset.
    print('\nFirst 5 rows of our Dataset:')
    print(data.head())
    #Print the Description of our Dataset.
    print('\nDescription of our Dataset:')
    print(data.describe())
    #Print the shape of our Dataset.
    print('\nThe shape of our Dataset:')
    print(data.shape)
    #Print the columns names of our Dataset.
    print('\nThe columns names of our Dataset:')
    print(data.columns)
    #Here we observe that we have 4 answers for 'sex' column and 6 for 'smoking'.
    #We need to clean and make our data clear and easy to understand by our model.
    print('\nHere we see how many unique values has each feature column:')
    print(data.nunique())
    #Print the unique values for sex and smoking feature.
    print('\nOver here we overview the unique values for sex and smoking feature:')
    print(data['sex'].unique())
    print(data['smoking'].unique())
    #We need to ckeck our dataset for any null values.
    print('\nHere we ckeck our dataset for any null values:')
    print(data.isnull().sum())
    return()

printing_our_dataset(data)

def make_replacements(data):
    # Here we replace the values that we need to make clear, to be understandable for our model.
    # with zero for not smoking and 1 for smoking.
    # and with zero for Female and 1 for Male.
    data = data.replace('yess', 'yes')
    data = data.replace('Yes', 'yes')
    data = data.replace('n', 'no')
    data = data.replace('No', 'no')
    data = data.replace('Woman', 'Female')
    data = data.replace('Man', 'Male')
    data = data.replace('yes', '1')
    data = data.replace('no', '0')
    data = data.replace('Female', '0')
    data = data.replace('Male', '1')
    data["sex"] = data["sex"].str.replace(',', '').astype(float)
    data["smoking"] = data["smoking"].str.replace(',', '').astype(float)
    data["diabetes"] = data["diabetes"].str.replace(',', '').astype(float)
    return(data)

data = make_replacements(data)

# We have checked our string values that we need to replace, in order to 
# clean our dataset, we have to check also our float values for outliers.
df1 = data.filter(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'])

def Outliers(df1):
    #calculate z-scores of `df1`
    z_scores = stats.zscore(df1)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df1 = df1[filtered_entries]

    df2 = new_df1.filter(['anaemia', 'creatinine_phosphokinase', 'ejection_fraction', 'DEATH_EVENT'])
    df3 = new_df1.filter(['diabetes', 'ejection_fraction', 'blood_pressure', 'DEATH_EVENT'])
    df4 = new_df1.filter(['ejection_fraction', 'platelets', 'serum_creatinine', 'DEATH_EVENT'])
    df5 = new_df1.filter(['ejection_fraction', 'serum_sodium', 'sex', 'DEATH_EVENT'])
    df6 = new_df1.filter(['ejection_fraction', 'smoking', 'time', 'DEATH_EVENT'])

    sns.set(style="ticks", color_codes=True)
    a = sns.pairplot(df2)
    b = sns.pairplot(df3)
    c = sns.pairplot(df4)
    d = sns.pairplot(df5)
    g = sns.pairplot(df6)
    plt.show()
    #(1)We can observe that age and anaemia doesn't affect each other.
    #(2)People with creatinine_phosphokinase more than 1000 have propability to not die in our dataset.
    #(3)The CPK(creatinine_phosphokinase) normal range for a male is between 39 – 308 U/L,
    #while in females the CPK normal range is between 26 – 192 U/L and we can observe that people with normal CPK die.
    #(4)A normal heart's ejection fraction may be between 50 and 70 percent and we can observe that only the 5% of the sample
    #has a normal percent number. Ejection fraction looks to affect the people from our dataset and takes the decision of their life.
    #(5)People with more than 4 serum_creatinine die.
    #(6)People with time more than 250 don't die.
    #(7)We observe that people with normal ejection fraction is very difficult to die from anaemia.

    #relationship analysis
    corelation = df2.corr()
    sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)
    plt.show()
    sns.relplot(x= 'ejection_fraction', y= 'anaemia', hue='DEATH_EVENT', data=new_df1)
    plt.show()
    sns.distplot(new_df1['ejection_fraction'])
    plt.show()
    sns.catplot(x='ejection_fraction', kind= 'box', data=new_df1)
    plt.show()
    return(new_df1)

new_df1 = Outliers(df1)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Part 2
# Perform the classification task and evaluate the performance of your models. How different
# parametres affect the performance of your model? What about feature importance?
print("\n Exercise 2 :")

# Specify features columns.
X = new_df1.drop(columns="DEATH_EVENT", axis=0)

# Specify target column.
y = new_df1["DEATH_EVENT"]

# Import required library for resampling.
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import ExtraTreesClassifier

# Instantiate Random Under Sampler.
rus = RandomUnderSampler(random_state=42)

# Perform random under sampling.
df_data, df_target = rus.fit_resample(X, y)

# Visualize new classes distributions.
sns.countplot(df_target).set_title('Balanced Data Set')
plt.show()

# Import required libraries for performance metrics.
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# Define dictionary with performance metrics.
scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}

# Import required libraries for machine learning classifiers.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Instantiate the machine learning classifiers.
log_model = LogisticRegression(max_iter=10000)
svc_model = LinearSVC(dual=False)
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()

# Define the models evaluation function.
def models_evaluation(X, y, folds):
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    
    # Perform cross-validation to each machine learning classifier.
    log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)
    dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores.
    models_scores_table = pd.DataFrame({'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision'].mean(),
                                                               log['test_recall'].mean(),
                                                               log['test_f1_score'].mean()],
                                       
                                      'Support Vector Classifier':[svc['test_accuracy'].mean(),
                                                                   svc['test_precision'].mean(),
                                                                   svc['test_recall'].mean(),
                                                                   svc['test_f1_score'].mean()],
                                       
                                      'Decision Tree':[dtr['test_accuracy'].mean(),
                                                       dtr['test_precision'].mean(),
                                                       dtr['test_recall'].mean(),
                                                       dtr['test_f1_score'].mean()],
                                       
                                      'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision'].mean(),
                                                       rfc['test_recall'].mean(),
                                                       rfc['test_f1_score'].mean()],
                                       
                                      'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                              gnb['test_precision'].mean(),
                                                              gnb['test_recall'].mean(),
                                                              gnb['test_f1_score'].mean()]},
                                      
                                      index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    # Add 'Best Score' column.
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    
    # Return models performance metrics scores data frame.
    return(models_scores_table)
  
# Run models_evaluation function and print it.
print('\nBest machine learning classifier for the parametre that we choose:')
# more options can be specified also.
#with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
print(models_evaluation(df_data, df_target, 5))

# When we run our code with different target parameters, we observe that our model have better 
# performance with other machine learning classifiers, depending on the target parameter each time.


def feature_importance(new_df1):
    #independent columns.
    X = new_df1.iloc[:,0:10] 
    #target column i.e price range.
    y = new_df1.iloc[:,-1]    
    model = ExtraTreesClassifier()
    model.fit(X,y)
    #use inbuilt class feature_importances of tree based classifiers.
    print(model.feature_importances_) 
    #plot graph of feature importances for better visualization.
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()
    return()

feature_importance(new_df1)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# Part 3
# Create a simple Neural Network from scratch to perform the classification task using only
# NumPy for building the model and the computational graph.
import warnings
warnings.filterwarnings("ignore") #suppress warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("\n Exercise 3 :")

# convert imput to numpy arrays.
X = new_df1.drop(columns=['DEATH_EVENT'])

y_label = new_df1['DEATH_EVENT'].values.reshape(X.shape[0], 1)

# split data into train and test set.
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

# standardize the dataset.
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

print('\nHere are the Shapes of the datasets, which will be fed in our Neural Network')
print(f"Shape of train set is {Xtrain.shape}")
print(f"Shape of test set is {Xtest.shape}")
print(f"Shape of train label is {ytrain.shape}")
print(f"Shape of test labels is {ytest.shape}")

class NeuralNet():
    '''
    A two layer neural network.
    '''
        
    def __init__(self, layers=[12,8,1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None
                
    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution.
        '''
        # Seed the random number generator.
        np.random.seed(1) 
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) 
        self.params['b1']  =np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) 
        self.params['b2'] = np.random.randn(self.layers[2],)
    
    def relu(self,Z):
        '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less 
        than zero are set to zero.
        '''
        return np.maximum(0,Z)

    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def eta(self, x):
      ETA = 0.0000000001
      return np.maximum(x, ETA)


    def sigmoid(self,Z):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1/(1+np.exp(-Z))

    def entropy_loss(self,y, yhat):
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        # clips value to avoid NaNs in log.
        yhat = self.eta(yhat) 
        yhat_inv = self.eta(yhat_inv) 
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss

    def forward_propagation(self):
        '''
        Performs the forward propagation.
        '''
        
        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y,yhat)

        # save calculated parameters.     
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat,loss

    def back_propagation(self,yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        y_inv = 1 - self.y
        yhat_inv = 1 - yhat

        dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(self.y, self.eta(yhat))
        dl_wrt_sig = yhat * (yhat_inv)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        #update the weights and bias.
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels.
        '''
        self.X = X
        self.y = y
        #initialize weights and bias.
        self.init_weights() 


        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        '''
        Predicts on a test data.
        '''
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred) 

    def acc(self, y, yhat):
        '''
        Calculates the accuracy between the predicted values and the truth labels.
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc


    def plot_loss(self):
        '''
        Plots the loss curve.
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()

# create the NN model with the default paramemeters.
nn = NeuralNet() 
# train the model.
nn.fit(Xtrain, ytrain) 
# print loss fuction.
nn.plot_loss()
plt.show()

# We have the following accuracies.
train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)

# print the accuracies.
print('\n NN accuracy with the default parameters:')
print("\nTrain accuracy is {}".format(nn.acc(ytrain, train_pred)))
print("Test accuracy is {}".format(nn.acc(ytest, test_pred)))

# create the NN model with more iterations.
nn = NeuralNet(layers=[12,8,1], learning_rate=0.01, iterations=500) 
# train the model.
nn.fit(Xtrain, ytrain) 
nn.plot_loss()
plt.show()

# We have the following accuracies.
train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)

# print the accuracies.
print('\n NN accuracy with different parameters such as learning_rate and iterations:')
print("\nTrain accuracy is {}".format(nn.acc(ytrain, train_pred)))
print("Test accuracy is {}".format(nn.acc(ytest, test_pred)))
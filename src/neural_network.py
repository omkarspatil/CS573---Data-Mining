import torch
import torch.autograd as autograd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy import stats
import molecule_reader
import read_csv_to_get_smiles
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


import warnings

#Made neural network dimensions global
idim = 2048  # input dimension
hdim1 = 1024  # hidden layer one dimension
hdim2 = 1024 # hidden layer two dimension
odim = 2  # output dimension

# This method creates a pytorch model
def create_model(idim, odim, hdim1, hdim2):
    model = torch.nn.Sequential(
            torch.nn.Linear(idim, hdim1),
            torch.nn.BatchNorm1d(hdim1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hdim1, hdim2),
            torch.nn.BatchNorm1d(hdim1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(hdim2, odim),
            torch.nn.Softmax()
            )
    return model

# This method trains a model with the given data
# Epoch is the number of training iterations
# lrate is the learning rate
def nn_train(train_x, train_y, model, epoch, lrate):
    inputs = []

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    X = train_x
    Y = train_y

    for itr in range(epoch):
        X_auto = autograd.Variable(torch.from_numpy(X), requires_grad=True).float()
        Y_auto = autograd.Variable(torch.from_numpy(Y), requires_grad=False).long()

        y_pred = model(X_auto)
        loss = loss_fn(y_pred, Y_auto)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

def nn_get_predictions(test_x, model):
    X = autograd.Variable(torch.from_numpy(test_x), requires_grad=False).float()
    y_pred = model(X)
    return y_pred.data.numpy()[:,1]

# Pass the trained model along with test data to this method to get accuracy
# The method return accuracy value
def nn_test(test_x, test_y, model):

    X = autograd.Variable(torch.from_numpy(test_x), requires_grad=False).float()
    y_pred = model(X)
    _, idx = torch.max(y_pred, 1)
    return (1.*np.count_nonzero((idx.data.numpy() == test_y).astype('uint8')))/len(test_y)


def normalize(data_list):
    # compute normalization parameter
    utter = np.concatenate(data_list, axis=0)
    mean  = np.mean(utter)
    utter -= mean
    std   = np.std(utter)
    utter /= std

    # normalize data
    for data in data_list:
        data -= mean
        data /= std

    return data_list

# Method reads training data and drops loan id field
def read_train_data(filename):
    f = open(filename, "r")
    x = []
    y = []
    content = f.readlines() 

    for i in range(1, len(content)):
        line = content[i]
        line.strip()
        line = line.split(",")
        y.append(line[len(line)-1])
        line = line[1:len(line)-1]
        x.append(line)


    x = np.array(x)
    x = x.astype(np.float)

    y = np.array(y)
    y = y.astype(np.int)
    return (x,y)


#This method performs the parameter tuning with discrete optimization
def param_tuning_with_discrete_optimization(k, X_train, y_train):
    df = X_train
    y = y_train

    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)

    candidate_epochs = [100, 150, 200]
    candidate_parameter_grid_learning_rate = [0.001, 0.01, 0.05, 0.1]

    best_accuracy_all_folds = 0
    best_params_overall = {}

    fold=1
    for train_index, test_index in skf.split(df, y):
        print("Parameter tuning fold", fold)
        df_train = df[train_index]
        y_train_k = y[train_index]
        X_test = df[test_index]
        y_test = y[test_index]

        #At this stage split the training split of this fold into 80-20
        #Run a grid search on the grid to find the best parameter, get the best parameter, train on the
        #original training split and check for accuracy, the goal is to return the best set of parameters
        #so found.
        X_train_pt, X_test_pt, y_train_pt, y_test_pt = train_test_split(df_train, y_train_k, random_state=42,test_size=0.2)

        best_accuracy = 0
        best_params ={}

        for learning_rate in candidate_parameter_grid_learning_rate:
            for epochs in candidate_epochs:
                print("Currently tuning with parameters :", learning_rate, epochs)
                trained_model = create_model(idim, odim, hdim1, hdim2)
                trained_model = nn_train(X_train_pt, y_train_pt, trained_model, epochs , learning_rate)
                accuracy_current_param_set = metrics.roc_auc_score(y_test_pt, nn_get_predictions(X_test_pt, trained_model))
                #Update the best parameters to the ones that give the best in the grid on training and testing 80-20
                if accuracy_current_param_set > best_accuracy:
                    best_accuracy = accuracy_current_param_set
                    best_params["learning_rate"] = learning_rate
                    best_params["epochs"] = epochs

        trained_model = create_model(idim, odim, hdim1, hdim2)
        trained_model = nn_train(df_train, y_train_k, trained_model, best_params["epochs"], best_params["learning_rate"])
        accuracy_fold = metrics.roc_auc_score(y_test, nn_get_predictions(X_test, trained_model))

        if accuracy_fold > best_accuracy_all_folds:
            best_accuracy_all_folds = accuracy_fold
            best_params_overall["epochs"] = best_params["epochs"]
            best_params_overall["learning_rate"] = best_params["learning_rate"]

        print("Best parameters for fold ", fold, ": ", best_params, " Optimal AUC so far: ", best_accuracy_all_folds)
        fold = fold + 1

    print("Best Accuracy over all folds", best_accuracy_all_folds,  "Best parameters overall ", best_params_overall)
    return best_params_overall

def kfold_cross_validation_nn(k, X_train, y_train, best_params=None):
    df = X_train
    y = y_train
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)

    accuracies = []
    fold = 1
    for train_index, test_index in skf.split(df, y):
        print("Parameter tuning fold", fold)
        df_train = df[train_index]
        y_train_k = y[train_index]
        X_test = df[test_index]
        y_test = y[test_index]

        trained_model = create_model(idim, odim, hdim1, hdim2)
        trained_model = nn_train(df_train, y_train_k, trained_model, best_params["epochs"], best_params["learning_rate"])
        accuracies.append(metrics.roc_auc_score(y_test, nn_get_predictions(X_test, trained_model)))
    print(accuracies)
    return accuracies

def plot_curves():
    X, y = read_training_data_as_fingerprints()
    best_params = param_tuning_with_discrete_optimization(10, X, y)
    #Uncomment the line below and comment the line above to skip the parameter tuning.
    #best_params =  {'learning_rate': 0.01, 'epochs': 10}

    aucs_train = []
    aucs_validation = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    training_volume_fractions = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    for volume_fraction in training_volume_fractions:
        print(volume_fraction)
        choice = np.random.choice(len(X_train), size=int(volume_fraction * len(X_train)), replace=False)
        X_train_n = X_train[choice]
        y_train_n = y_train[choice]

        trained_model = create_model(idim, odim, hdim1, hdim2)
        trained_model = nn_train(X_train_n, y_train_n, trained_model, best_params["epochs"],
                                 best_params["learning_rate"])
        probas_ = nn_get_predictions(X_train_n, trained_model)

        auc_train = roc_auc_score(y_train_n, probas_)

        probas_v = nn_get_predictions(X_test, trained_model)

        auc_validation = roc_auc_score(y_test, probas_v)

        print(auc_train)
        aucs_train.append(auc_train)
        aucs_validation.append(auc_validation)
        print(auc_validation)

    plot_learning_curve(aucs_train, aucs_validation, "NN")
    plot_roc_learning_curve(best_params, "NN")

def read_training_data_as_fingerprints():
    data = read_csv_to_get_smiles.read_csv()
    X, y = molecule_reader.generate_training_vectors(data, False)
    X = np.array(X)
    y = np.array(y)
    return X,y

def plot_learning_curve(training_aucs, testing_aucs, plot_name):
    training_volume_fractions = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    fig_nn = plt.figure()
    plt.style.use('seaborn')
    plt.plot(training_volume_fractions, training_aucs, 'o-')
    plt.plot(training_volume_fractions, testing_aucs, 'o-')
    plt.legend(['Training AUC Scores', 'Testing AUC Scores'], loc='lower right')
    plt.xlabel('Training Data Volume')
    plt.ylabel('AUC')
    plt.title('Learning curve for model ' + plot_name)
    plt.xticks(training_volume_fractions)
    #plt.show()
    plt.ylim(0)
    plt.savefig(plot_name+"_learning.png", dpi = fig_nn.dpi)
    plt.close(fig_nn)

def plot_roc_learning_curve(best_params,plot_name):
    X, y = read_training_data_as_fingerprints()
    cv = StratifiedKFold(n_splits=10)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig_nn = plt.figure()

    i = 0
    for train, test in cv.split(X, y):
        trained_model = create_model(idim, odim, hdim1, hdim2)
        trained_model = nn_train(X[train], y[train], trained_model, best_params["epochs"],
                                 best_params["learning_rate"])
        probas_ = nn_get_predictions(X[test], trained_model)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.style.use('seaborn')
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC for fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random Guessing', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for model '+plot_name)
    plt.legend(loc="lower right")
    plt.savefig(plot_name + "_roc.png", dpi = fig_nn.dpi)
    plt.close(fig_nn)

def k_fold_run():
    X, y = read_training_data_as_fingerprints()
    best_params = param_tuning_with_discrete_optimization(10, X, y)
    #Uncomment the line below and comment the line above to skip the parameter tuning.
    #best_params =  {'learning_rate': 0.01, 'epochs': 10}
    print(kfold_cross_validation_nn(10, X, y, best_params))

k_fold_run()
plot_curves()




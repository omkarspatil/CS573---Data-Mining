import molecule_reader
import read_csv_to_get_smiles
from sklearn import linear_model
from sklearn import cross_validation
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import itertools
from scipy import interp
from scipy import stats

np.set_printoptions(threshold=np.nan)

def read_training_data_as_fingerprints():
    data = read_csv_to_get_smiles.read_csv()
    X, y = molecule_reader.generate_training_vectors(data, False)
    X = np.array(X)
    y = np.array(y)
    return X,y

def RF_train(X, y,X_test,y_test, parameters):
    model = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=50, max_features='auto', **parameters)
    model.fit(X, y)
    ac_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return ac_test

def XG_train(X, y, X_test,y_test, parameters):
    model = XGBClassifier(**parameters)
    model.fit(X, y)
    ac_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return ac_test

def LR_train(X,y, X_test,y_test ,parameters):
    estimator = linear_model.LogisticRegression(**parameters, class_weight="balanced")
    estimator.fit(X, y)
    ac_test = roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1])
    return ac_test

#Main hashmap containing model descriptions including parameter tuning grids
map_train_test_function_to_possible_parameters =\
    {
    "LR": {
        "train_test_function": LR_train,
        "parameter_choices":
            {
                "C": np.logspace(-20, 20, num=100)
            },
        "model": linear_model.LogisticRegression,
        "default_parameters" : {'class_weight':"balanced"}
    },
    "RF" : {
        "train_test_function": RF_train,
        "parameter_choices":
        {
        "n_estimators" : [50, 100, 150, 200],
        "min_samples_leaf" : [1, 5, 10],
        "max_depth" : [2, 3, 4, 10, 20, 30, 40, 45]
        },
        "model":RandomForestClassifier,
        "default_parameters" : {'oob_score':True, 'n_jobs':-1, 'random_state':50, 'max_features':'auto'}
    },
    "XG": {
        "train_test_function": XG_train,
        "parameter_choices":
            {
                "eta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "max_depth": [2, 3, 4, 5, 6, 7]
            },
        "model":XGBClassifier,
        "default_parameters" : {}
    }
}

def hypotheses_testing(accuraciesA, accuraciesB):
    hypothesis_test_results = stats.ttest_rel(accuraciesA, accuraciesB)
    print(hypothesis_test_results)

def test_xg_against_rest():
    accuracies = test_after_tuning()
    hypotheses_testing(accuracies['XG'], accuracies['LR'])
    hypotheses_testing(accuracies['XG'], accuracies['RF'])

def test_after_tuning():
    X, y = read_training_data_as_fingerprints()
    best_parameters_all = param_tuning_with_discrete_optimization(10,X,y)
    '''best_parameters_all = {
        "LR":  {'C': 0.01519911082952927},
        "RF" : {'n_estimators': 500, 'min_samples_leaf': 1, 'max_depth': 7},
        "XG" : {'eta': 0.1, 'max_depth': 2}
    }'''
    accuracy_lists = k_fold_cross_validation(10, X, y, best_parameters_all)

    for model, model_map in map_train_test_function_to_possible_parameters.items():
        print("Statistics for model ",model)
        print(np.mean(accuracy_lists[model]))
        print(np.std(accuracy_lists[model]))

    #print(accuracy_lists)
    return accuracy_lists


def k_fold_cross_validation(k, X_train, y_train,best_parameters):
    df = X_train
    y = y_train
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    best_accuracy_all_folds = {}
    accuracies_all_folds = {}

    for model, model_map in map_train_test_function_to_possible_parameters.items():
        accuracies_all_folds[model] =[]

    fold = 1
    for train_index, test_index in skf.split(df, y):
        print("Cross-validation fold", fold)
        df_train = df[train_index]
        y_train_k = y[train_index]
        X_test = df[test_index]
        y_test = y[test_index]

        for model, model_map in map_train_test_function_to_possible_parameters.items():
            best_accuracy_all_folds[model] =0
            accuracy_fold = model_map["train_test_function"](df_train, y_train_k, X_test, y_test, best_parameters[model])
            accuracies_all_folds[model].append(accuracy_fold)
        fold = fold + 1
    return accuracies_all_folds


#This method performs the parameter tuning with discrete optimization
def param_tuning_with_discrete_optimization(k, X_train, y_train):
    df = X_train
    y = y_train

    skf=StratifiedKFold(n_splits=k, random_state=None, shuffle=False)



    #count_best_params_overall = {}
    best_accuracy_all_folds = {}
    best_params_overall ={}
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

        best_params = {}
        best_accuracy ={}
        for model, model_map in map_train_test_function_to_possible_parameters.items():
            best_accuracy[model] = 0
            param_names = []
            params_combined_lists =[]

            for param_name, param_list in model_map["parameter_choices"].items():
                param_names.append(param_name)
                params_combined_lists.append(param_list)

            for parameter_combination in itertools.product(*params_combined_lists):
                zipped = zip(param_names, parameter_combination)
                dict_params = dict(zipped)
                #print("Running for model ", model, ":", dict_params)
                accuracy_current_param_set= model_map["train_test_function"](X_train_pt,y_train_pt, X_test_pt,
                                                                              y_test_pt, dict_params)
                if accuracy_current_param_set > best_accuracy[model]:
                    best_accuracy[model] = accuracy_current_param_set
                    best_params[model] = dict_params

            accuracy_fold = model_map["train_test_function"](df_train, y_train_k, X_test,y_test, best_params[model])

            '''if model in count_best_params_overall:
                if tuple(best_params[model].values()) in count_best_params_overall[model]:
                    count_best_params_overall[model][tuple(best_params[model].values())] += 1
                else:
                    count_best_params_overall[model][tuple(best_params[model].values())] = 1
            else:
                count_best_params_overall[model] ={}
                count_best_params_overall[model][tuple(best_params[model].values())] = 1'''


            if accuracy_fold > best_accuracy_all_folds[model]:
                best_accuracy_all_folds[model] = accuracy_fold
                best_params_overall[model] = best_params[model]

            print("Best parameters till fold ", fold, ": ", "Model" , model, best_params_overall[model]," AUC :", accuracy_fold)
        fold = fold + 1
    print("Best Accuracy over all folds", best_accuracy_all_folds,  "Best parameters overall ", best_params_overall)

    #print(count_best_params_overall)
    return best_params_overall


def test_parameter_tuning():
    X, y = read_training_data_as_fingerprints()
    param_tuning_with_discrete_optimization(10, X, y)


def plot_all_learning_curves():
    X, y = read_training_data_as_fingerprints()
    best_parameters_all = param_tuning_with_discrete_optimization(10,X,y)

    '''best_parameters_all = {
        "LR":  {'C': 0.01519911082952927},
        "RF" : {'n_estimators': 500, 'min_samples_leaf': 1, 'max_depth': 7},
        "XG" : {'eta': 0.1, 'max_depth': 2}
    }'''

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    for model, best_parameters in best_parameters_all.items():
        aucs_train =[]
        aucs_validation=[]
        training_volume_fractions = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        for volume_fraction in training_volume_fractions:
            print(volume_fraction)
            choice = np.random.choice(len(X_train), size=int(volume_fraction * len(X_train)), replace=False)
            X_train_n = X_train[choice]
            y_train_n = y_train[choice]
            auc_train = map_train_test_function_to_possible_parameters[model]['train_test_function'](X_train_n, y_train_n, X_train_n, y_train_n, best_parameters)
            auc_validation = map_train_test_function_to_possible_parameters[model]['train_test_function'](X_train_n, y_train_n, X_test, y_test, best_parameters)
            print(auc_train)
            aucs_train.append(auc_train)
            aucs_validation.append(auc_validation)
            print(auc_validation)
        plot_learning_curve(aucs_train, aucs_validation, model)

def plot_learning_curve(training_aucs, testing_aucs, plot_name):
    training_volume_fractions = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    fig_nn = plt.figure()
    plt.style.use('seaborn')
    plt.plot(training_volume_fractions, training_aucs, 'o-')
    plt.plot(training_volume_fractions, testing_aucs, 'o-')
    plt.legend(['Training AUC Scores', 'Testing AUC Scores'], loc='lower right')
    plt.xlabel('Training Data Volume')
    plt.xlabel('AUC')
    plt.title('Learning curve for model ' + plot_name)
    plt.xticks(training_volume_fractions)
    #plt.show()
    plt.ylim(0)
    plt.savefig(plot_name+"_learning.png", dpi = fig_nn.dpi)
    plt.close(fig_nn)

def plot_all_roc_curves():

    '''best_parameters_all = {
        "LR":  {'C': 0.01519911082952927},
        "RF" : {'n_estimators': 500, 'min_samples_leaf': 1, 'max_depth': 7},
        "XG" : {'eta': 0.1, 'max_depth': 2}
    }'''

    best_parameters_all ={
     'LR': {'C': 0.005994842503189421},
     'RF': {'n_estimators': 150, 'min_samples_leaf': 1, 'max_depth': 10},
     'XG': {'eta': 0.1, 'max_depth': 2}
    }

    for model, best_parameters in best_parameters_all.items():
        model_to_train = map_train_test_function_to_possible_parameters[model]['model'](
            **map_train_test_function_to_possible_parameters[model]['default_parameters'], **best_parameters)
        plot_roc_learning_curve(model_to_train, model)

def plot_roc_learning_curve(model, plot_name):
    X, y = read_training_data_as_fingerprints()

    cv = StratifiedKFold(n_splits=10)
    classifier = model

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig_nn = plt.figure()

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
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

def test():

    data = read_csv_to_get_smiles.read_test_csv()
    X_test = molecule_reader.generate_training_vectors(data, True)
    #y_predict = estimator.predict_proba(X_test)[:,1]
    #y_predict = estimator.predict(X_test)
    #y_true = np.ones(np.array(X_test).shape[0])
    #print(y_true.shape)
    #print(roc_auc_score(y_true, estimator.predict_proba(X_test)[:,1]))
    #print(estimator.score(X_test, y_true))
    #print(data['index'])
    #print(data.shape, y_predict.shape)
    #final_output = np.column_stack((data['index'], y_predict))
    #print(final_output)
    #np.savetxt("out_final.csv", final_output, delimiter=",", fmt=["%d", "%.9f"])

#Performs parameter tuning
test_parameter_tuning()
#Tests accuracy of trained models with best hyper-parameters
test_after_tuning()
#Performs multiple hypothesis testing
test_xg_against_rest()
#Plots all learning curves
plot_all_learning_curves()
#Plots all roc curves
plot_all_roc_curves()


#Utility function used to write the submission csvs.
#test()



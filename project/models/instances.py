# Instances of all supported classifiers
# If desired classifier is not listed add by creating instance
# with unique name and append it to __instances list
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, \
    LassoLars
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from models.Model import Model
from sklearn import exceptions

from utils import print_progress


class Instances(object):
    decisionTree0 = DecisionTreeClassifier(max_depth=1)
    decisionTree1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    decisionTree2 = DecisionTreeClassifier(splitter='random', max_depth=1)
    decisionTree3 = DecisionTreeClassifier(splitter='random', criterion='entropy', max_depth=1)
    decisionTree4 = DecisionTreeClassifier(max_features='sqrt', max_depth=1)
    decisionTree5 = DecisionTreeClassifier(max_features='log2', max_depth=1)
    decisionTree6 = DecisionTreeClassifier(criterion='entropy', max_features='log2', max_depth=1)
    decisionTree7 = DecisionTreeClassifier(splitter='random', max_features='log2', max_depth=1)
    decisionTree8 = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', max_depth=1)
    decisionTree9 = DecisionTreeClassifier(splitter='random', max_features='sqrt', max_depth=1)

    naiveBayes0 = GaussianNB()
    naiveBayes1 = GaussianNB(var_smoothing=1)
    naiveBayes2 = GaussianNB(var_smoothing=0)
    naiveBayes3 = GaussianNB(var_smoothing=1e9)

    svm0 = SVC()
    svm1 = SVC(gamma='auto')
    svm2 = SVC(C=1e-2)
    svm3 = SVC(C=1e2)
    svm4 = SVC(kernel='poly')
    svm5 = SVC(kernel='poly', gamma='auto')
    svm6 = SVC(kernel='sigmoid')
    svm7 = SVC(kernel='sigmoid', gamma='auto')
    svm8 = SVC(kernel='linear')
    svm9 = SVC(kernel='linear')

    stochasticGradient0 = SGDClassifier()
    stochasticGradient1 = SGDClassifier(loss='log')
    stochasticGradient2 = SGDClassifier(loss='modified_huber')
    stochasticGradient3 = SGDClassifier(loss='squared_hinge')
    stochasticGradient4 = SGDClassifier(loss='perceptron')
    stochasticGradient5 = SGDClassifier(penalty='elasticnet')
    stochasticGradient6 = SGDClassifier(loss='log', penalty='elasticnet')
    stochasticGradient7 = SGDClassifier(loss='modified_huber', penalty='elasticnet')
    stochasticGradient8 = SGDClassifier(loss='squared_hinge', penalty='elasticnet')
    stochasticGradient9 = SGDClassifier(loss='perceptron', penalty='elasticnet')

    kNeighbors0 = KNeighborsClassifier()
    kNeighbors1 = KNeighborsClassifier(n_neighbors=2)
    kNeighbors2 = KNeighborsClassifier(n_neighbors=10)
    kNeighbors3 = KNeighborsClassifier(weights='distance')
    kNeighbors4 = KNeighborsClassifier(algorithm='ball_tree')
    kNeighbors5 = KNeighborsClassifier(algorithm='kd_tree')
    kNeighbors6 = KNeighborsClassifier(algorithm='brute')
    kNeighbors7 = KNeighborsClassifier(algorithm='ball_tree', weights='distance')
    kNeighbors8 = KNeighborsClassifier(algorithm='kd_tree', weights='distance')
    kNeighbors9 = KNeighborsClassifier(algorithm='brute', weights='distance')

    gaussianProcess0 = GaussianProcessClassifier()
    gaussianProcess1 = GaussianProcessClassifier(warm_start=True)
    gaussianProcess2 = GaussianProcessClassifier(n_restarts_optimizer=1)
    gaussianProcess3 = GaussianProcessClassifier(n_restarts_optimizer=1, warm_start=True)
    gaussianProcess4 = GaussianProcessClassifier(max_iter_predict=1000)
    gaussianProcess5 = GaussianProcessClassifier(warm_start=True, max_iter_predict=1000)
    gaussianProcess6 = GaussianProcessClassifier(n_restarts_optimizer=1, max_iter_predict=1000)
    gaussianProcess7 = GaussianProcessClassifier(n_restarts_optimizer=1, warm_start=True, max_iter_predict=1000)
    gaussianProcess8 = GaussianProcessClassifier(max_iter_predict=10)
    gaussianProcess9 = GaussianProcessClassifier(n_restarts_optimizer=5)

    logisticRegression0 = LogisticRegression()
    logisticRegression3 = LogisticRegression(penalty='none')
    logisticRegression2 = LogisticRegression(solver='liblinear', penalty='l1')
    logisticRegression4 = LogisticRegression(solver='newton-cg')
    logisticRegression1 = LogisticRegression(solver='saga', penalty='l1')
    logisticRegression5 = LogisticRegression(solver='saga', penalty='none')
    logisticRegression6 = LogisticRegression(solver='saga')
    logisticRegression7 = LogisticRegression(solver='liblinear', penalty='l2')
    #logisticRegression8 = LogisticRegression(solver='sag', penalty='none')
    #logisticRegression9 = LogisticRegression(solver='sag')

    ridgeClassification0 = RidgeClassifier(copy_X=True)
    #ridgeClassification1 = RidgeClassifier(copy_X=True, tol=1, solver='cholesky')
    #ridgeClassification2 = RidgeClassifier(copy_X=True, tol=1e-1, solver='cholesky')
    ridgeClassification3 = RidgeClassifier(copy_X=True, tol=1, solver='sparse_cg')
    ridgeClassification4 = RidgeClassifier(copy_X=True, tol=1e-1, solver='sparse_cg')
    #ridgeClassification5 = RidgeClassifier(copy_X=True, solver='cholesky')
    #ridgeClassification6 = RidgeClassifier(copy_X=True, solver='sag')
    ridgeClassification7 = RidgeClassifier(copy_X=True, solver='saga')
    #ridgeClassification8 = RidgeClassifier(copy_X=True, solver='sag', random_state=0)
    ridgeClassification9 = RidgeClassifier(copy_X=True, solver='saga', random_state=42)

    passiveAggressive0 = PassiveAggressiveClassifier()
    passiveAggressive1 = PassiveAggressiveClassifier(C=10)
    passiveAggressive2 = PassiveAggressiveClassifier(C=10, loss='hinge')
    passiveAggressive3 = PassiveAggressiveClassifier(loss='squared_hinge')
    passiveAggressive4 = PassiveAggressiveClassifier(early_stopping=True)
    passiveAggressive5 = PassiveAggressiveClassifier(C=10, early_stopping=True)
    passiveAggressive6 = PassiveAggressiveClassifier(C=10, loss='hinge', early_stopping=True)
    passiveAggressive7 = PassiveAggressiveClassifier(loss='squared_hinge', early_stopping=True)
    passiveAggressive8 = PassiveAggressiveClassifier(n_iter_no_change=1)

    quadraticDiscriminantAnalysis0 = QuadraticDiscriminantAnalysis()
    quadraticDiscriminantAnalysis1 = QuadraticDiscriminantAnalysis(store_covariance=True)
    quadraticDiscriminantAnalysis2 = QuadraticDiscriminantAnalysis(tol=1.0e-1)
    quadraticDiscriminantAnalysis3 = QuadraticDiscriminantAnalysis(tol=1.0e-1, store_covariance=True)

    linearDiscriminantAnalysis0 = LinearDiscriminantAnalysis()
    linearDiscriminantAnalysis1 = LinearDiscriminantAnalysis(store_covariance=True)
    linearDiscriminantAnalysis2 = LinearDiscriminantAnalysis(n_components=1)
    linearDiscriminantAnalysis3 = LinearDiscriminantAnalysis(solver='lsqr')
    linearDiscriminantAnalysis4 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    linearDiscriminantAnalysis5 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=1.0)
    linearDiscriminantAnalysis6 = LinearDiscriminantAnalysis(solver='eigen')
    linearDiscriminantAnalysis7 = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    linearDiscriminantAnalysis8 = LinearDiscriminantAnalysis(solver='eigen', shrinkage=1.0)

    lasso0 = LassoLars(copy_X=True)
    lasso1 = LassoLars(copy_X=True, alpha=0.5)
    lasso2 = LassoLars(copy_X=True, fit_intercept=False)
    lasso3 = LassoLars(copy_X=True, alpha=0.5, fit_intercept=False)
    lasso4 = LassoLars(copy_X=True, fit_path=False)
    lasso5 = LassoLars(copy_X=True, alpha=0.5, fit_path=False)
    lasso6 = LassoLars(copy_X=True, fit_intercept=False, fit_path=False)
    lasso7 = LassoLars(copy_X=True, alpha=0.5, fit_intercept=False, fit_path=False)
    lasso8 = LassoLars(copy_X=True, jitter=1)
    lasso9 = LassoLars(copy_X=True, fit_intercept=False, positive=True)
    lasso10 = LassoLars(copy_X=True, alpha=0.5, fit_intercept=False, positive=True)

    __instance = [gaussianProcess8,
                  svm8,
                  svm9,
                  stochasticGradient0,
                  stochasticGradient1,
                  stochasticGradient2,
                  stochasticGradient6,
                  stochasticGradient7,
                  gaussianProcess4,
                  gaussianProcess5,
                  kNeighbors1,
                  kNeighbors2,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  decisionTree0,
                  ]

    __instances = [decisionTree0,
                   decisionTree1,
                   decisionTree2,
                   decisionTree3,
                   decisionTree4,
                   decisionTree5,
                   decisionTree6,
                   decisionTree7,
                   decisionTree8,
                   decisionTree9,
                   naiveBayes0,
                   naiveBayes1,
                   naiveBayes2,
                   naiveBayes3,
                   svm0,
                   svm1,
                   svm2,
                   svm3,
                   svm4,
                   svm5,
                   svm6,
                   svm7,
                   svm8,
                   svm9,
                   stochasticGradient0,
                   stochasticGradient1,
                   stochasticGradient2,
                   stochasticGradient3,
                   stochasticGradient4,
                   stochasticGradient5,
                   stochasticGradient6,
                   stochasticGradient7,
                   stochasticGradient8,
                   stochasticGradient9,
                   kNeighbors0,
                   kNeighbors1,
                   kNeighbors2,
                   kNeighbors3,
                   kNeighbors4,
                   kNeighbors5,
                   kNeighbors6,
                   kNeighbors7,
                   kNeighbors8,
                   kNeighbors9,
                   gaussianProcess0,
                   gaussianProcess1,
                   gaussianProcess2,
                   gaussianProcess3,
                   gaussianProcess4,
                   gaussianProcess5,
                   gaussianProcess6,
                   gaussianProcess7,
                   gaussianProcess8,
                   gaussianProcess9,
                   logisticRegression0,
                   logisticRegression1,
                   logisticRegression2,
                   logisticRegression3,
                   logisticRegression4,
                   logisticRegression5,
                   logisticRegression6,
                   logisticRegression7,
                   #logisticRegression8,
                   #logisticRegression9,
                   #ridgeClassification0,
                   #ridgeClassification1,
                   #ridgeClassification2,
                   #ridgeClassification3,
                   #ridgeClassification4,
                   #ridgeClassification5,
                   #ridgeClassification6,
                   #ridgeClassification7,
                   #ridgeClassification8,
                   #ridgeClassification9,
                   passiveAggressive0,
                   passiveAggressive1,
                   passiveAggressive2,
                   passiveAggressive3,
                   passiveAggressive4,
                   passiveAggressive5,
                   passiveAggressive6,
                   passiveAggressive7,
                   passiveAggressive8,
                   quadraticDiscriminantAnalysis0,
                   quadraticDiscriminantAnalysis1,
                   quadraticDiscriminantAnalysis2,
                   quadraticDiscriminantAnalysis3,
                   linearDiscriminantAnalysis0,
                   linearDiscriminantAnalysis1,
                   linearDiscriminantAnalysis2,
                   linearDiscriminantAnalysis3,
                   linearDiscriminantAnalysis4,
                   linearDiscriminantAnalysis5,
                   linearDiscriminantAnalysis6,
                   linearDiscriminantAnalysis7,
                   linearDiscriminantAnalysis8,
                   #lasso0,
                   #lasso1,
                   #lasso2,
                   #lasso3,
                   #lasso4,
                   #lasso5,
                   #lasso6,
                   #lasso7,
                   #lasso8,
                   #lasso9,
                   #lasso10,
                   ]

    __trained_classifiers = []
    __predictions_classifiers = []
    __scores = []

    @property
    def trained_classifiers(self):
        return self.__trained_classifiers

    @property
    def predictions_arr(self):
        return self.__predictions_classifiers

    @property
    def instances(self):
        return self.__instances

    @property
    def scores(self):
        return self.__scores

    def trainClassifiers(self, X, y):
        for i, instance in enumerate(self.__instances):
            try:
                print_progress(i + 1, len(self.__instances), "Training")
                trained_model = instance.fit(X, y)
            except exceptions.FitFailedWarning as e:
                print("An error occurred while training classifier. Omitting.", e)
            else:
                self.__trained_classifiers.append(trained_model)
        print('')

    def predictClassifiers(self, X):
        model = Model()
        for i, instance in enumerate(self.__trained_classifiers):
            try:
                print_progress(i + 1, len(self.__trained_classifiers), "Predicting")
                predictions = instance.predict(X)
            except exceptions.NotFittedError as e:
                print("An error occurred while estimating classes. Omitting.", e)
            else:
                try:
                    self.__scores.append(model.calcScore(predictions, verify=False))
                except ValueError as e:
                    print(" Couldn't calculate score. Omitting.", e)
                else:
                    self.__predictions_classifiers.append(predictions)
        print('')

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# from sklearn.base import clone
# from abc import ABCMeta, abstractmethod
from ..tree.tree import (WeightedDecisionTreeClassifier, 
                         WeightedDecisionTreeRegressor)
from ..utils import get_rf_tree_data
import numpy as np
from sklearn.tree import _tree as sk_tree

class RandomForestClassifierWithWeights(RandomForestClassifier):
    def __init__(self, 
                 n_estimators=100,
                 *,
                 criterion="squared_error",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=1.0,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
    @property
    def n_paths(self):
        if not hasattr(self, "estimators_"):
            return 0
        out = 0
        for tree in self.estimators_:
            out += np.sum(tree.tree_.feature == sk_tree.TREE_UNDEFINED) #out += np.sum(tree.tree_.feature == -2)
        return out
    def fit(self, X, y, sample_weight=None, feature_weight=None):
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # Create a new custom tree with any custom parameters
            tree = WeightedDecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                feature_weight=feature_weight  # <- custom param
            )
            # Fit tree
            tree.fit(X, y, sample_weight=sample_weight)
            self.estimators_.append(tree)
        return self # super(RandomForestRegressorWithWeights, self).fit(X, y, sample_weight)

class RandomForestRegressorWithWeights(RandomForestRegressor):
    def __init__(self, 
                 n_estimators=100,
                 *,
                 criterion="squared_error",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=1.0,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
    @property
    def n_paths(self):
        if not hasattr(self, "estimators_"):
            return 0
        out = 0
        for tree in self.estimators_:
            out += np.sum(tree.tree_.feature == sk_tree.TREE_UNDEFINED) #out += np.sum(tree.tree_.feature == -2)
        return out
    def fit(self, X, y, sample_weight=None, feature_weight=None):
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # Create a new custom tree with any custom parameters
            tree = WeightedDecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                feature_weight=feature_weight  # <- custom param
            )
            # Fit tree
            tree.fit(X, y, sample_weight=sample_weight)
            self.estimators_.append(tree)
        return self # super(RandomForestRegressorWithWeights, self).fit(X, y, sample_weight)
        
class wrf(RandomForestClassifierWithWeights):
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5, 
            keep_record = True, X_test = None, y_test = None, sample_weight_test=None):
        self.all_rf_weights = dict()
        if keep_record:
            self.all_K_iter_rf_data = dict()
            assert X_test is not None, 'X_test should not be None when keep_record'
            assert y_test is not None, 'y_test should not be None when keep_record'
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = feature_weight
                self.all_rf_weights['rf_weights{}'.format(k)] = feature_importances
                
            # fit weighted RF
            # fit the classifier
            super().fit( #super(wrf, self).fit(
                    X=X,
                    y=y,
                    sample_weight=sample_weight,
                    feature_weight=feature_importances)
            
            # Update feature weights using the
            # new feature importance score
            feature_importances = self.feature_importances_
            self.all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances
            if keep_record:
                self.all_K_iter_rf_data["rf_iter{}".format(k+1)] = get_rf_tree_data(
                        rf=self,
                        X_train=X,
                        X_test=X_test,
                        y_test=y_test,
                        sample_weight_test=sample_weight_test)
        return self

#Eric: Doesn't lfook like much is changed here, as it looks like no significant differences between regressor and classifier
#       come back to later to make sure there really is no difference                
class wrf_reg(RandomForestRegressorWithWeights): # Hue: change the name so that it does not clash with the first one.
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5, 
            keep_record = True, X_test = None, y_test = None):
        self.all_rf_weights = dict()
        if keep_record:
            self.all_K_iter_rf_data = dict()
            assert X_test is not None, 'X_test should not be None when keep_record'
            assert y_test is not None, 'y_test should not be None when keep_record'
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = feature_weight
                self.all_rf_weights['rf_weights{}'.format(k)] = feature_importances
                
            # fit weighted RF
            # fit the regressor
            super().fit( #super(wrf_reg, self).fit(
                    X=X,
                    y=y,
                    feature_weight=feature_importances)
            
            # Update feature weights using the
            # new feature importance score
            feature_importances = self.feature_importances_
            self.all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances
            if keep_record:
                self.all_K_iter_rf_data["rf_iter{}".format(k+1)] = get_rf_tree_data(
                        rf=self,
                        X_train=X,
                        X_test=X_test,
                        y_test=y_test) 
        return self

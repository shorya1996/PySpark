import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce

# Define a replacement for OrderedSet
class UniqueList:
    def __init__(self, iterable=None):
        self.items = []
        self.item_set = set()
        if iterable:
            for item in iterable:
                self.add(item)

    def add(self, item):
        if item not in self.item_set:
            self.items.append(item)
            self.item_set.add(item)

    def update(self, iterable):
        for item in iterable:
            self.add(item)

    def __iter__(self):
        return iter(self.items)

    def __contains__(self, item):
        return item in self.item_set

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

# Define supporting classes
class RuleCondition:
    def __init__(self, feature_index, threshold, operator, support, feature_name=None, is_binary=False):
        self.feature_index = feature_index
        self.operator = operator
        self.support = support
        self.feature_name = feature_name
        self.is_binary = is_binary
        self.threshold = 1 if is_binary and operator == "<=" else 0 if is_binary else threshold

    def transform(self, X):
        if self.is_binary:
            return 1 * (X[:, self.feature_index] == self.threshold)  # Binary match check
        else:
            return 1 * (X[:, self.feature_index] <= self.threshold) if self.operator == "<=" else 1 * (X[:, self.feature_index] > self.threshold)

    def __str__(self):
        return f"{self.feature_name if self.feature_name else self.feature_index} {self.operator} {self.threshold}"


class Rule:
    def __init__(self, rule_conditions, prediction_value):
        self.conditions = UniqueList(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        self.prediction_value = prediction_value

    def transform(self, X):
        return reduce(lambda x, y: x * y, [condition.transform(X) for condition in self.conditions])

    def __str__(self):
        return " & ".join([str(cond) for cond in self.conditions])


class RuleEnsemble:
    def __init__(self, tree_list, feature_names=None, categorical_features=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.rules = UniqueList()
        self._extract_rules()
        self.rules = list(self.rules)

    def _extract_rules(self):
        for tree in self.tree_list:
            rules = self._extract_rules_from_tree(tree[0].tree_)
            self.rules.update(rules)

    def _extract_rules_from_tree(self, tree):
        rules = UniqueList()
        binary_columns = [self.feature_names.index(feature) for feature in self.feature_names if feature in self.categorical_features]

        def traverse_nodes(node_id=0, operator=None, threshold=None, feature=None, conditions=[]):
            if node_id != 0:
                feature_name = self.feature_names[feature] if self.feature_names else feature
                is_binary = feature in binary_columns
                new_conditions = conditions + [
                    RuleCondition(
                        feature, threshold, operator,
                        tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                        feature_name, is_binary=is_binary
                    )
                ]
            else:
                new_conditions = []
            if tree.children_left[node_id] != tree.children_right[node_id]:  # not a leaf
                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                traverse_nodes(tree.children_left[node_id], "<=", threshold, feature, new_conditions)
                traverse_nodes(tree.children_right[node_id], ">", threshold, feature, new_conditions)
            else:  # leaf node
                new_rule = Rule(new_conditions, tree.value[node_id][0][0])
                rules.update([new_rule])
        traverse_nodes()
        return rules

    def transform(self, X):
        transformed_data = np.array([rule.transform(X) for rule in self.rules]).T
        return transformed_data


class RuleFit(BaseEstimator, TransformerMixin):
    def __init__(self, tree_size=4, max_rules=2000, memory_par=0.01, rfmode="classify", model_type="rl", random_state=None, max_iter=1000):
        self.tree_size = tree_size
        self.max_rules = max_rules
        self.memory_par = memory_par
        self.rfmode = rfmode
        self.model_type = model_type
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y, feature_names=None):
        if isinstance(X, pd.DataFrame):
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            print(f"Categorical features identified: {categorical_features}")

            X = pd.get_dummies(X, drop_first=True)
            self.feature_names = list(X.columns)
            print(f"Updated feature names after one-hot encoding: {self.feature_names}")
        else:
            categorical_features = []
            self.feature_names = list(feature_names) if feature_names is not None else ["feature_" + str(i) for i in range(X.shape[1])]

        self.categorical_features = categorical_features
        X = X.values if isinstance(X, pd.DataFrame) else X

        self.tree_generator = RandomForestClassifier(
            n_estimators=int(np.ceil(self.max_rules / self.tree_size)),
            max_leaf_nodes=self.tree_size,
            random_state=self.random_state
        )
        self.tree_generator.fit(X, y)

        tree_list = [[tree] for tree in self.tree_generator.estimators_]
        self.rule_ensemble = RuleEnsemble(
            tree_list=tree_list,
            feature_names=self.feature_names,
            categorical_features=self.categorical_features
        )

        self.lscv = LogisticRegressionCV(cv=3, penalty="l1", max_iter=self.max_iter, solver="liblinear").fit(self.rule_ensemble.transform(X), y)
        self.X = X
        self.y = y
        return self

    def get_rules(self, min_support=0.01, min_accuracy=0.7):
        rule_data = []
        for rule in self.rule_ensemble.rules:
            fraud_rate_captured, captured_cases = self._compute_fraud_rate(rule)
            rule_data.append({
                'rule': str(rule),
                'coef': self.lscv.coef_[0][self.rule_ensemble.rules.index(rule)],
                'support': rule.support,
                'fraud_rate': fraud_rate_captured,
                'captured_cases': captured_cases,
                'length': len(rule.conditions)
            })
        rules_df = pd.DataFrame(rule_data)
        return rules_df[(rules_df['coef'] != 0) & (rules_df['support'] >= min_support)]

    def _compute_fraud_rate(self, rule):
        rule_predictions = rule.transform(self.X)
        total_cases = len(self.y)
        total_fraud_cases = np.sum(self.y == 1)
        captured_fraud_cases = np.sum((rule_predictions == 1) & (self.y == 1))
        fraud_rate_captured = captured_fraud_cases / total_cases if total_cases > 0 else 0.0
        return fraud_rate_captured, captured_fraud_cases

#Plot
import matplotlib.pyplot as plt

#Math packages
import pandas as pd
import math

import numpy as np
import itertools

class Task(object):
    def __init__(self, dislikes, likes, leadership, learning, language):
        self.dislikes = dislikes
        self.likes = likes
        self.leadership_features = leadership
        self.learning_features = learning
        self.language_features = language
        n_students = set([len(likes["value"]), len(dislikes["value"]), len(self.leadership_features["value"]), len(self.learning_features["value"]), len(self.language_features["value"])])
        assert len(n_students) == 1, "the input of the task is inconsistent: # students appeared in the input - {}".format(n_students)
        self.n_students = list(n_students)[0]
        self.features = {"leadership": self.leadership_features, 
            "learning": self.learning_features, 
            "language": self.language_features}
        self.weights = {"leadership": 1/3,
            "learning": 1/3,
            "language": 1/3
        }

    def check_like(self, student1_id, student2_id):
        return True if self.likes["value"][student1_id][student2_id] else False
    def check_mutual_like(self, student1_id, student2_id):
        return self.check_like(student1_id, student2_id) and self.check_like(student2_id, student1_id)
    def check_dislike(self, student1_id, student2_id):
        return True if self.dislikes["value"][student1_id][student2_id] else False
    def check_any_dislike(self, *student_ids):
        flag = False
        for x, y in itertools.product(student_ids, student_ids):
            if x != y and self.check_dislike(x, y):
                print("group {} has a dislike pair: {} hates {}".format(tuple(student_ids), x, y))
                flag = True
        return flag
    
    def check_constraints_group(self, group, soft_on_number=False):
        flag = True
        if not soft_on_number and (len(group) > 4 or len(group) < 3): 
            flag = False
            print("group {} has invalid number of people: {}".format(tuple(group), len(group)))
        if self.check_any_dislike(*group):
            flag = False
        return flag

    def check_constraints(self, assignment):
        flag = True
        for group in assignment:
            if self.check_constraints_group(group):
                flag = False
        return flag

    def _diversity_objective(self, group, feature):
        return np.array([feature["value"][student_id] for student_id in group]).max(axis=0).sum()

    def get_personal_features(self, student_id):
        return {feature_name: list(map(lambda x: x[0], filter(lambda x: x[1], {feature["description"][i]: feature["value"][student_id][i] 
            for i in range(feature["value"].shape[1])}.items())))
                for feature_name, feature in self.features.items()}

    def get_diversity_score_group(self, group):
        return {feature_name:self._diversity_objective(group, feature) 
                for feature_name, feature in self.features.items()}

    def get_diversity_score(self, assignment):
        score = []
        for group in assignment:
            score.append(self.get_diversity_score_group(group))
        return score

    def get_total_score(self, assignment):
        score = self.get_diversity_score(assignment)
        return np.mean([sum(list(map(lambda x: self.weights[x[0]] * x[1], ind_score.items()))) 
            for ind_score in score]) 

    def print_features(self, task):
        for group in task:
            print(group)
            for student_id in group:
                print("{}: {}".format(student_id, self.get_personal_features(student_id)))
            print("diversity scores: {}".format(self.get_diversity_score_group(group)))           


class Assignment(object):
    def __init__(self, group_ids, n_groups):
        self.n_groups = n_groups
        self.group_ids = np.array(group_ids)
        assert np.all(self.group_ids < n_groups), "invalid assignments"
        self.groups = [[] for _ in range(n_groups)]
        for student_id, group_id in enumerate(group_ids): self.groups[group_id].append(student_id)
        for group in self.groups: 
            group.sort()

    def __iter__(self):
        for group in self.groups: yield group

    def __repr__(self):
        return "Assignment({}, {})".format(self.group_ids, self.n_groups)

class Search(object):
    def __init__(self, task):
        self.task = task

    def go(self):
        pass

def random_assignment(task):
    group_ids = []
    n_groups = 0
    curr_num = 0 
    for _ in range(task.n_students):
        if curr_num == 4: 
            curr_num = 0 
            n_groups += 1
        group_ids.append(n_groups)
        curr_num += 1
    if curr_num != 0: n_groups += 1
    return Assignment(group_ids, n_groups)

def parse_feature(df):
    return {"value": np.array(df.values, dtype=np.bool), "description": list(df.columns.values)}

languageDF = pd.read_excel('Frances_Material/MaximizingTeamDiversityData.xlsx', sheet_name='Language').drop('Languages', axis=1).drop("Students")
leaderDF = pd.read_excel('Frances_Material/MaximizingTeamDiversityData.xlsx', sheet_name='Leadership').drop('Lead. Styles', axis=1).drop("Students")
learningDF = pd.read_excel('Frances_Material/MaximizingTeamDiversityData.xlsx', sheet_name='Learning').drop('Learn. Styles', axis=1).drop("Students")
likesDF = pd.read_excel('Frances_Material/MaximizingTeamDiversityData.xlsx', sheet_name='Like')
dislikesDF = pd.read_excel('Frances_Material/MaximizingTeamDiversityData.xlsx', sheet_name='Dislikes')

dfs = {"language": languageDF, "leadership": leaderDF, "learning": learningDF, "likes": likesDF, "dislikes": dislikesDF}
dfs = {feature_name: parse_feature(feature_df) for feature_name, feature_df in dfs.items()}
task = Task(**dfs)
assignment = random_assignment(task)

print(assignment)
print(task.check_constraints(assignment))
print(task.get_total_score(assignment))
task.print_features(assignment)

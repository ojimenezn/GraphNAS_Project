import json
from copy import deepcopy

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Analysis:
    def __init__(self, link, type, dataset, nr_layers):
        self.link = link
        self.type = type
        self.dataset = dataset
        self.nr_layers = nr_layers

        if self.type == 'macro':
            self.df = self.return_df_macro()
        elif self.type == 'micro':
            self.hp_list = ['Learning Rate', 'Dropout', 'L2', 'Hidden Layer Size']
            self.act_list= [['in_1','in_2'],['f_1'],['Activation Function'], ['Aggregation Function']]
            self.act_list_labels = ['Network Structure', 'Layer Types', 'Activation Function', 'Aggregation Function']
            self.df = self.return_df_micro()
        else:
            raise Exception("marco/micro")

    def return_df_macro(self):
        f = self.read_file_macro()
        df = pd.DataFrame(f, columns=['actions', 'val'])
        df[['layer_1', 'agg_1', 'acc_1', 'att_1', 'hidd_dim_1', 'layer_2', 'agg_2', 'acc_2', 'att_2',
            'hidd_dim_2']] = pd.DataFrame(df.actions.tolist(), index=df.index)
        return df

    def return_df_micro(self):
        f = self.read_file_micro()
        df = pd.DataFrame(f, columns=['actions', 'hyper_param', 'val'])
        df[['in_1', 'f_1', 'in_2', 'f_2', 'Activation Function', 'Aggregation Function']] = pd.DataFrame(df.actions.tolist(), index=df.index)
        df[self.hp_list] = pd.DataFrame(df.hyper_param.tolist(),
                                                                                   index=df.index)
        return df

    def read_file_macro(self):
        results = []
        with open(self.link, "r") as f:
            lines = f.readlines()

        for line in lines:
            actions = line[:line.index(";")]

            actions_dataform = str(actions).strip("'<>() ").replace('\'', '\"')
            actions_struct = {}
            actions_struct = json.loads(actions_dataform)

            val_score = line.split(";")[-1]
            val_dataform = str(val_score).strip("'<>() ").replace('\'', '\"')
            val_struct = {}
            val_struct = json.loads(val_dataform)

            results.append((actions_struct, val_struct))

        return results

    def read_file_micro(self):
        results = []
        with open(self.link, "r") as f:
            lines = f.readlines()

        for line in lines:
            actions = line[:line.index(";")]

            dataform = str(actions).strip("'<>() ").replace('\'', '\"')
            struct = {}
            struct = json.loads(dataform)
            actions_struct = struct['action']
            hyper_param_struct = struct['hyper_param']

            val_score = line.split(";")[-1]
            val_dataform = str(val_score).strip("'<>() ").replace('\'', '\"')
            val_struct = {}
            val_struct = json.loads(val_dataform)

            results.append((actions_struct, hyper_param_struct, val_struct))

        return results

    def analise_over_action(self, action_name):
        new_df = pd.DataFrame(self.df[[action_name, 'val']].groupby(action_name).mean())
        return new_df

    def analise_over_combinations_of_actions(self, actions):
        l = deepcopy(actions)
        actions.append('val')
        print(l)
        df = self.df[actions].groupby(l).mean()
        return df

    def explore_hyper_param(self):
        if self.type != 'micro':
            raise Exception("Sorry, wrong type")

        figure, axis = plt.subplots(2, 2, figsize=(8, 8))
        for i in range(len(self.hp_list)):
            A = self.analise_over_action(self.hp_list[i])
            x = A.index.array.to_numpy()
            y = A['val'].array.to_numpy()
            x_pos = np.arange(len(y))

            a = int(i/2)
            b = i%2

            axis[a][b].bar(x_pos, y)
            axis[a][b].set_ylabel("Validation Accuracy")
            axis[a][b].set_xlabel(self.hp_list[i])
            axis[a][b].set_ylim(0.45, 0.9)
            axis[a][b].set_xticks(x_pos, labels=x, minor=False)
            axis[a][b].tick_params(axis='both', which='major', labelsize=5)
        figure.tight_layout()

        name = self.dataset + "_" + self.type + '_hyperparam.jpg'
        plt.savefig('../plots/'+name,dpi=150)

    def explore_micro_architectures(self):
        if self.type != 'micro':
            raise Exception("Sorry, wrong type")

        figure, axis = plt.subplots(2, 2, figsize=(8, 8))
        #print(self.df)

        for i in range(len(self.act_list)):
            A = self.analise_over_combinations_of_actions(self.act_list[i])
            if 'f_1' in self.act_list[i]:
                A = A.sort_values('val', ascending=False).head(20)
            if(len(self.act_list[i])>2):
                x = []
                codes = A.index.levels
                for j in range(len(A.index.codes[0])):
                    x.append(str(A.index.codes[0][j])+"/"+str(A.index.codes[1][j]))
            else:
                x = A.index.array.to_numpy()
            y = A['val'].array.to_numpy()
            x_pos = np.arange(len(y))

            a = int(i/2)
            b = i%2

            axis[a][b].bar(x_pos, y)
            axis[a][b].set_ylabel("Validation Accuracy")
            axis[a][b].set_xlabel(self.act_list_labels[i])
            axis[a][b].set_ylim(0.45, 0.9)
            axis[a][b].set_xticks(x_pos, labels=x, minor=False)
            axis[a][b].tick_params(axis='both', which='major', labelsize=5)
        figure.tight_layout()

        name = self.dataset + "_" + self.type + '_all_functions.jpg'
        plt.savefig('../plots/' + name, dpi=150)


cora_micro = Analysis('../Citeseer_microsub_manager_logger_file_1673002125.8153458.txt', dataset='Citeseer', type='micro', nr_layers = 2)
cora_micro.explore_micro_architectures()

"""
@author: siddhant agarwal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# !pip install factor_analyzer
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import statsmodels.api as sm
# !pip install pingouin
import pingouin as pg
import itertools
from collections import defaultdict

class factor_analysis:
    def __init__(self, data):
        self.data = data
        
    def data_subset(self, question):
        self.ques = question
        cols =[]
        for i in self.data.columns:
            if question in i:
                cols.append(i)    
        if len(cols) == 0:
            print('Enter a valid question')
        else:
            self.data_sub = self.data[cols]
            
    def drop_columns(self):
        # If you wish to drop columns based on a condition
        drop_cols = []
        for i in self.data_sub.columns:
            a = self.data_sub.groupby(i, as_index = False).count()
            try:
                if a[a[i] == 'your condition'].iloc[0, 1]/len(self.data_sub)*100 > 'your condition':
                    drop_cols.append(i)
            except:
                continue
        self.data_sub = self.data_sub.drop(drop_cols, axis = 1)
        if len(drop_cols) == 0:
            print("No column has more than 'your condition' not applicable entries")
            return True
        elif len(drop_cols) == self.data_sub.shape[1]:
            print("All columns have more than 'your condition' not applicable entries")
            return False
        else:
            print("Columns with more than 'your condition' not applicable entries are: ", drop_cols)
            return True
        
    def remove_notapplicable(self):
        ind = []
        for i in self.data_sub.columns:
            ind.append(self.data_sub[self.data_sub[i] == 'your condition'].index)
        rows_to_drop = list(itertools.chain.from_iterable(ind))
        rows_to_drop = list(dict.fromkeys(rows_to_drop))
        self.data_sub_filter = self.data_sub.drop(rows_to_drop) 
        self.data_sub_filter = self.data_sub_filter.dropna() #Change this code for dealing with null entries
        # Ensuring the N:p ratio to be more than 20
        if len(self.data_sub_filter) <= 20*self.data_sub_filter.shape[1]:
            print("Respondents to items ratio less than 20:1")
            print("Number of respondents %d and number of items %d" %(len(self.data_sub_filter), self.data_sub_filter.shape[1]))
            return False
        else:
            print("Respondents per items is: ", len(self.data_sub_filter)//self.data_sub_filter.shape[1])
            print("Number of respondents %d and number of items %d" %(len(self.data_sub_filter), self.data_sub_filter.shape[1]))
            return True
    
    def factor_test(self):
        chi_square_value,p_value=calculate_bartlett_sphericity(self.data_sub_filter)
        kmo_all,kmo_model=calculate_kmo(self.data_sub_filter)
        if (p_value >= 0.05) or (kmo_model <= 0.5):
            print("Data is not suitable for factor analysis as p-value for Bartlett's Test of Sphericity is {} and Kaiser-Meyer-Olkin Measure is {}" .format(p_value, np.round(kmo_model,3)))
            return False
        else:
            print("Data is suitable for factor analysis as p-value for Bartlett's Test of Sphericity is {} and Kaiser-Meyer-Olkin Measure is {}" .format(p_value, np.round(kmo_model,3)))
            return True
            
    def distribution_test(self):
        fig = sm.qqplot(self.data_sub_filter, fit=True, line="45")
        plt.show()
        self.normal = pg.multivariate_normality(self.data_sub_filter, alpha=.05)[-1]
        if self.normal:
            print("Data follows multivariate normal distribution thus extraction method used is Maximum Likelihood")
        else:
            print("Data does not follow multivariate normal distribution thus extraction method used is PAF")
            
    def init_factor(self):
        if self.normal:
            fa = FactorAnalyzer(method = 'ml', rotation = None)
        else:
            fa = FactorAnalyzer(method = 'principal', rotation = None)
        fa.fit(self.data_sub_filter)
        # Check Eigenvalues
        ev, v = fa.get_eigenvalues()
        # Create scree plot
        plt.plot(range(1,self.data_sub_filter.shape[1]+1),ev, marker = 'o')
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.grid()
        plt.show()
        
    def factors_analysis(self, no_of_factors):
        if self.normal:
            fa = FactorAnalyzer(method = 'ml')
        else:
            fa = FactorAnalyzer(method = 'principal')
        fa.set_params(n_factors= no_of_factors, rotation='oblimin')
        fa.fit(self.data_sub_filter)
        loadings = fa.loadings_
        self.factor_df = pd.DataFrame(loadings, columns = [self.ques+' Factor' + str(i+1) for i in range(no_of_factors)], index = self.data_sub_filter.columns)
        self.factor_df = pd.concat([self.factor_df, pd.DataFrame(fa.get_communalities(), columns = ['communalities'], index = self.data_sub_filter.columns)], axis = 1)
        drop_index = []
        for i in self.factor_df.index:
            if self.factor_df.loc[i, 'communalities'] < .4:
                drop_index.append(i)
        if len(drop_index) == 0:      
            print("No item has communalities less than 0.4")
            self.variance_score(fa)
            return True
        elif len(drop_index) == len(self.factor_df):
            print("All items have communalities less than 0.4")
            return False
        else:
            print("Items with communalities less than 0.4 are: ", drop_index)
            self.data_sub_filter = self.data_sub_filter.drop(drop_index, axis = 1)
            self.factors_analysis(no_of_factors)
            return True
            
    def variance_score(self, fa):
        
#         self.factor_df = self.factor_df.drop(drop_index, axis = 0)
        self.df = pd.DataFrame(fa.get_factor_variance(), columns = self.factor_df.columns[:-1], index = ['SS Loadings', 'Proportion Var', 'Cumulative Var'])
#         self.scores = pd.DataFrame(fa.transform(self.data_sub), columns = self.factor_df.columns[:-1])

        dic = {}
        for i in self.factor_df.index:
            s = self.factor_df.loc[i, self.factor_df.columns[:-1]]
            if any(s == s.abs().max()): 
                is_max = s == s.abs().max()
            else:
                is_max = -s == s.abs().max()

            for ind, v in enumerate(is_max):
                if v:
                    dic[i] = is_max.index[ind]

        self.new_dict = defaultdict(list)
        for k, v in dic.items():
            self.new_dict[v].append(k)
            
        self.score_df = pd.DataFrame(columns = self.factor_df.columns[:-1])
        for i in self.score_df.columns:
            self.score_df[i] = self.data_sub[self.new_dict[i]].mean(axis = 1)
            print('Cronbach alpha value for '+ i + ' is:', np.round(pg.cronbach_alpha(self.data_sub_filter[self.new_dict[i]])[0],2))
            
        return True
        
    def highlight_max(self, s):
        '''
        highlight the maximum in a Series yellow.
        '''
        if any(s == s.abs().max()): 
            is_max = s == s.abs().max()
        else:
            is_max = -s == s.abs().max()
        return ['background-color: yellow' if v else '' for v in is_max]
    
    def Print(self):
        display(self.factor_df.style.apply(self.highlight_max, axis = 1))
        print(self.df)
#         self.factor_df = self.factor_df.append(self.df)
#         self.factor_df.fillna('', inplace = True)
#         print(self.factor_df)
        return self.factor_df.style.apply(self.highlight_max, axis = 1), self.df, self.score_df
    
fa = factor_analysis('data to be input')
# Enter the question number to input
print('The question whose factor analysis needs to be conducted')
fa.data_subset(str(input()))
drp = fa.drop_columns()
if drp:
    rm = fa.remove_notapplicable()
    if rm:
        ft = fa.factor_test()
        if ft:
            fa.distribution_test()
            fa.init_factor()
            print('The number of factors that needs to be extracted')
            ret = fa.factors_analysis(int(input()))
            if ret:
                style, variance, scores = fa.Print()
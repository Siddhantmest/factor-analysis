# Factor Analysis for survey data

`Factor_analysis.py` will help you to perform factor analysis on a subset of your survey data. The extraction method is based on the distribution of the data, the code automatically verifies the distribution and chooses an extraction method accordingly. The rotation method used is oblique as according to various research papers if the factors are truly uncorrelated, orthogonal, and oblique rotation produce nearly identical results. Communality constraint is added to the code where it automatically drops the columns/features/question if the communality score is less than .4 (obtained from various research papers) and then again performs factor analysis on the remaining data. After performing the factor analysis, you will obtain the reliability score (Cronbach alpha value) corresponding to each factor. The code outputs a factor dataframe highlighting the features that heavily load a particular factor. Factor scores are output by averaging the features which heavily load that factor.
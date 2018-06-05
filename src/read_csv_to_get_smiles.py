import numpy as np

def read_csv():
    data = np.genfromtxt("molecule_training.csv",delimiter=',',names=['index', 'Maximum Degree','Minimum Degree',	'Molecular Weight', 'Number of H-Bond Donors',	'Number of Rings',	'Number of Rotatable Bonds',	'Polar Surface Area',	'inchi_key', 'Graph'	, 'smiles', 'target'], dtype=None, comments='##', encoding=None)
    return data

def read_test_csv():
    data = np.genfromtxt("test_new.csv",delimiter=',',names=['index', 'Maximum Degree','Minimum Degree',	'Molecular Weight', 'Number of H-Bond Donors',	'Number of Rings',	'Number of Rotatable Bonds',	'Polar Surface Area',	'inchi_key', 'Graph'	,'smiles'], dtype=None, comments='##', encoding=None)
    return data

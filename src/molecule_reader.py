from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import SimilarityMaps
import numpy as np

#Helper function to convert to Morgan type fingerprint in Numpy Array
def genFP(mol,Dummy=-1):
     fp = SimilarityMaps.GetMorganFingerprint(mol)
     fp_vect = np.zeros((1,))
     DataStructs.ConvertToNumpyArray(fp, fp_vect)
     return fp_vect

def generate_training_vectors(data, testing):
     mols = []
     X = []
     y = []
     for record in data:
      try:
         mol = Chem.MolFromSmiles(record[10])
         if type(mol) != type(None):
           fp_vect = genFP(mol)
           mols.append([mol])
           if testing:
               #print(fp_vect.shape, record[1].shape)
               #print(fp_vect.shape)
               '''fp_vect = np.append(fp_vect, record[1])
               fp_vect = np.append(fp_vect, record[2])
               fp_vect = np.append(fp_vect, record[3])
               fp_vect = np.append(fp_vect, record[4])
               fp_vect = np.append(fp_vect, record[5])
               fp_vect = np.append(fp_vect, record[6])
               fp_vect = np.append(fp_vect, record[7])'''
               X.append(fp_vect)
           else:
               #print(fp_vect.shape)
               #print(fp_vect.shape, record[1].shape)
               '''fp_vect = np.append(fp_vect, record[1])
               fp_vect = np.append(fp_vect, record[2])
               fp_vect = np.append(fp_vect, record[3])
               fp_vect = np.append(fp_vect, record[4])
               fp_vect = np.append(fp_vect, record[5])
               fp_vect = np.append(fp_vect, record[6])
               fp_vect = np.append(fp_vect, record[7])'''
               X.append(fp_vect)
               y.append(record[11])
      except:
         print("Failed for CAS: %s"%record[8])
     #See how succesful the conversions were
     print("Imported smiles %s"%len(data))
     print("Converted smiles %s"%len(mols))

     if not testing:
        #print(X[0].shape, len(X))
        return X,y
     else:
        print(X[0].shape, len(X))
        return X


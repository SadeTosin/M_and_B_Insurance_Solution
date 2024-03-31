import pickle
import sys
sys.path.append(r'C:\Users\FOLASADE\OneDrive\Desktop\Projects\Insurance')

# Open the .pkl file in binary mode for reading
with open(r"C:\Users\FOLASADE\OneDrive\Desktop\Projects\Insurance\artifacts\proprocessor.pkl", "rb") as f:

    # Load the object from the file
    loaded_object = pickle.load(f)

# Now you can use the loaded_object
print(loaded_object)

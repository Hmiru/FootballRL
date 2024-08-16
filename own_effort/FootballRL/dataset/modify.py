import pandas as pd
import numpy as np

# Load the CSV file
data = pd.read_csv("total_state_data_with_paths.csv", index_col=0)

# Define the string to replace
change_word = "/camin1/Hmiru/dataset/"

# Check the initial state of the 'state_path' column
print("Before replacement:")
print(data['state_path'].head())

# Apply the replacement
data['state_path'] = data['state_path'].apply(lambda x: x.replace("..//dataset//", change_word))

# Check the state of the 'state_path' column after replacement
print("After replacement:")
print(data['state_path'].head())

# Save the modified DataFrame to a new CSV file
data.to_csv("total_state_data_with_paths.csv")
if __name__ == "__main__":
    new_data=pd.read_csv("total_state_data_with_paths.csv", index_col=0)
    test=np.load(new_data['state_path'][0])
    print(test.shape)
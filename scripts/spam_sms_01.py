import os
import sys

data_path = "/workspaces/codespaces-jupyter/data"
data_filename = "spam_sms_mini.txt"

if __name__ == "__main__":
    # List the files in the data directory
    filename = os.listdir(data_path)
    print("Files in {data_path}:")
    for i in filename:
        print(i)

# Check for existence of data file
if os.path.exists(os.path.join(data_path, data_filename)):
    print(f"Data file {data_filename} exists.")
else:
    print(f"Data path {data_path} and {data_filename} are not found.")
    sys.exit(0)

# Read the data file
data_file = os.path.join(data_path, data_filename)
data = list()
label = list()
with open(data_file, mode='r', encoding='utf-8') as myfile:
        for s in myfile.readlines():
            print()
            processed_str = s.split(sep=" ", maxsplit=1)
            label.append(processed_str[0])
            data.append(processed_str[1])
            print(processed_str[0], processed_str[1][:30])
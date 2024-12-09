import csv

def save_csv(data, file_name=f"temp", header=None, numbering=False):
    file_path = f"Saves/{file_name}.csv"
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        if header: 
            writer.writerow(header)

        if numbering:
            for index, item in enumerate(data, start=1):  # Start numbering from 1
                writer.writerow([index, item])
        else:
            for item in data:
                writer.writerow([item])

    print(f"Data saved to {file_path}")

def load_csv(file_name):
    file_path = f"Saves/{file_name}.csv"
    loaded_data = []
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            loaded_data.append(row[0])
        
    return loaded_data
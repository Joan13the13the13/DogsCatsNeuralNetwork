import cv2
import os
import pandas as pd
import csv
import torch
import torch.nn.functional as F

"""
Breeds dict: assigns unique identifier to each class.
"""
breeds = {
    'Abyssinian': 0,
    'american_bulldog': 1,
    'american_pit_bull_terrier': 2,
    'basset_hound': 3,
    'beagle': 4,
    'Bengal': 5,
    'Birman': 6,
    'Bombay': 7,
    'boxer': 8,
    'British_Shorthair': 9,
    'chihuahua': 10,
    'Egyptian_Mau': 11,
    'english_cocker_spaniel': 12,
    'english_setter': 13,
    'german_shorthaired': 14,
    'great_pyrenees': 15,
    'havanese': 16,
    'japanese_chin': 17,
    'keeshond': 18,
    'leonberger': 19,
    'Maine_Coon': 20,
    'miniature_pinscher': 21,
    'newfoundland': 22,
    'Persian': 23,
    'pomeranian': 24,
    'pug': 25,
    'Ragdoll': 26,
    'Russian_Blue': 27,
    'saint_bernard': 28,
    'samoyed': 29,
    'scottish_terrier': 30,
    'shiba_inu': 31,
    'Siamese': 32,
    'Sphynx': 33,
    'staffordshire_bull_terrier': 34,
    'wheaten_terrier': 35,
    'yorkshire_terrier': 36
}

"""
Get_min_dimensions: gets minimum height and width from a specified path.
Args: folder_path
Returns: minimum width,height
"""
def get_min_dimensions(folder_path):
    min_width = float('inf')
    min_height = float('inf')

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(image_path)
                height, width, _ = img.shape
                min_width = min(min_width, width)
                min_height = min(min_height, height)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return min_width, min_height

"""
Check_duplicates: allos to find duplicates in csv given it's path.
Args: csv_path
Returns: duplicated ids list.
"""
def check_duplicates(csv_path):
    existing_ids = set()
    # List to store duplicate ids
    duplicate_ids = []

    # Read existing CSV file
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            id_value = row['id']
            if id_value in existing_ids:
                duplicate_ids.append(id_value)
            else:
                existing_ids.add(id_value)

    return duplicate_ids
"""
Remove_duplicates: function that allows to remove duplicates from a given input csv path.
Then writes it's new content into another path. Can overwrite the same file.
Args: input_csv_path,output_csv_path
Returns: None, writes new content to output_csv path.
"""
def remove_duplicates(input_csv_path, output_csv_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_path)

    # Drop duplicate rows based on all columns
    df_unique = df.drop_duplicates()

    # Write the unique rows to a new CSV file
    df_unique.to_csv(output_csv_path, index=False)

"""
Extract_breed: extracts breed from a given filename.
Args: gets filename string.
Returns: breed name without underscores or numbers.
"""
def extract_breed(value):
    # Split by underscores, join all parts except the last one, and remove the trailing number
    parts = value.split('_')
    return '_'.join(parts[:-1])

"""
Process_file: given a file path, extract filename and then it's breed.
After that, get each breed id and 0,1 if cat/dog (cat filenames start with capital letter).
We finally write down that as a row in output csv.
Args: file_path, output_csv
Returns: None, writes down a row in given csv path.
"""
def process_file(file_path, output_csv):
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Initialize default values
    id_value = filename
    class_value = breeds.get(extract_breed(id_value))
    species_value = 1 if filename[0].isupper() else 2


    # Write to CSV
    with open(output_csv, mode='a', newline='') as csv_file:
        fieldnames = ['id', 'class', 'species']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header only if the file is empty
        if os.stat(output_csv).st_size == 0:
            writer.writeheader()

        writer.writerow({'id': id_value, 'class': class_value, 'species': species_value})


"""
ExtractValue function: allows to extract a determined column from a row in 
.csv that matches given filename. 

Usage example: 

    extractValue("y_train.csv",'Abyss','weight')
    
This sentence would extract weight column from y_train.csv that
has column id='Abyss'.
"""
def extractValue(csv,filename,column):
    matching_row = csv.loc[csv['id'] == filename]

    # Check if a matching row is found
    if not matching_row.empty:
        value = matching_row[column].item()
        return value
    else:
        # Handle the case where no matching row is found
        return None

"""
Train function: allows to train a determined model with specified parameters.
Args: takes a model, device, train data loader and optimizer.
Returns: whole epoch loss
"""
def train(model, device, train_loader, optimizer):
    model.train()

    loss_v = 0

    for batch_idx, (data, target) in enumerate(train_loader):
    
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.squeeze(output)
        loss=F.cross_entropy(output, target, reduction='mean')
        loss.backward()
        optimizer.step() 
        loss_v += loss.item()

    loss_v /= len(train_loader.dataset)
 
    return loss_v


"""
Test function: allows to test a model with specified parameters.
Args: model,device and test_loader.
Returns: whole epoch loss
"""
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.squeeze(output)
            test_loss += F.cross_entropy(output, target, reduction='mean') 
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss

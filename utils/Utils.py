import cv2
import os
import pandas as pd
import csv
import torch
import torch.nn.functional as F


breeds = {
    'Abyssinian': 1,
    'american_bulldog': 2,
    'american_pit_bull_terrier': 3,
    'basset_hound': 4,
    'beagle': 5,
    'Bengal': 6,
    'Birman': 7,
    'Bombay': 8,
    'boxer': 9,
    'British_Shorthair': 10,
    'chihuahua': 11,
    'Egyptian_Mau': 12,
    'english_cocker_spaniel': 13,
    'english_setter': 14,
    'german_shorthaired': 15,
    'great_pyrenees': 16,
    'havanese': 17,
    'japanese_chin': 18,
    'keeshond': 19,
    'leonberger': 20,
    'Maine_Coon': 21,
    'miniature_pinscher': 22,
    'newfoundland': 23,
    'Persian': 24,
    'pomeranian': 25,
    'pug': 26,
    'Ragdoll': 27,
    'Russian_Blue': 28,
    'saint_bernard': 29,
    'samoyed': 30,
    'scottish_terrier': 31,
    'shiba_inu': 32,
    'Siamese': 33,
    'Sphynx': 34,
    'staffordshire_bull_terrier': 35,
    'wheaten_terrier': 36,
    'yorkshire_terrier': 37
}
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

def check_duplicates(csv_path):
    # Set to store existing ids for duplicate check
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

def remove_duplicates(input_csv_path, output_csv_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv_path)

    # Drop duplicate rows based on all columns
    df_unique = df.drop_duplicates()

    # Write the unique rows to a new CSV file
    df_unique.to_csv(output_csv_path, index=False)

def extract_breed(value):
    # Split by underscores, join all parts except the last one, and remove the trailing number
    parts = value.split('_')
    return '_'.join(parts[:-1])

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



def extractValue(csv,filename,column,train=True):
    folder_path = 'data/train/' if train else 'data/test/'

    matching_row = csv.loc[csv['id'] == filename]

    # Check if a matching row is found
    if not matching_row.empty:
        value = matching_row[column].item()
        return value
    else:
        # Handle the case where no matching row is found
        return None

"""
Train function: allows to train a determined model with specified parameters
Args: takes a model, device, train data loader, optimizer and current epoch
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

import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_names", default="all", type=str)
argparser.add_argument("--dataset_path", default="DATASET/odinw", type=str)
args = argparser.parse_args()

root = "https://huggingface.co/GLIPModel/GLIP/resolve/main/odinw_35"

all_datasets = ["AerialMaritimeDrone", "AmericanSignLanguageLetters", "Aquarium", "BCCD", "ChessPieces", "CottontailRabbits", "DroneControl", "EgoHands", "HardHatWorkers", "MaskWearing", "MountainDewCommercial", "NorthAmericaMushrooms", "OxfordPets", "PKLot", "Packages", "PascalVOC", "Raccoon", "ShellfishOpenImages", "ThermalCheetah", "UnoCards", "VehiclesOpenImages", "WildfireSmoke", "boggleBoards", "brackishUnderwater", "dice", "openPoetryVision", "pistols", "plantdoc", "pothole", "selfdrivingCar", "thermalDogsAndPeople", "vector", "websiteScreenshots"]

datasets_to_download = []
if args.dataset_names == "all":
    datasets_to_download = all_datasets
else:
    datasets_to_download = args.dataset_names.split(",")

if not os.path.exists(args.dataset_path):
    os.makedirs(args.dataset_path)

for dataset in datasets_to_download:
    if dataset in all_datasets:
        
        dataset_dir = os.path.join(args.dataset_path, dataset)
        
        if os.path.exists(dataset_dir):
            print(f"[{dataset}] 文件夹已存在，跳过下载。")
            continue

        print("------------------------------------------------")
        print(f"Processing dataset: {dataset}")
        
        download_url = f"{root}/{dataset}.zip?download=true"
        save_path = os.path.join(args.dataset_path, f"{dataset}.zip")
        
        os.system(f"wget -N '{download_url}' -O {save_path}")
        
        print("Unzipping...")
        os.system(f"unzip -q {save_path} -d {args.dataset_path}")
        
        print("Cleaning up zip file...")
        os.system(f"rm {save_path}")
            
    else:
        print(f"Dataset not found in list: {dataset}")
dataset_prefixes=('AerialMaritimeDrone', 'Aquarium', 'CottontailRabbits', 'EgoHands', 'NorthAmericaMushrooms', 'Packages', 'PascalVOC', 'pistols', 'pothole', 'Raccoon', 'ShellfishOpenImages', 'thermalDogsAndPeople', 'VehiclesOpenImages')
datasets=(
    {   
        "metainfo": {
            "name": "AerialMaritimeDrone",
            "classes": ('boat', 'car', 'dock', 'jetski', 'lift'),
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/AerialMaritimeDrone",
        "ann_file": "large/test/annotations_without_background.json",
        "data_prefix": {
            "img": "large/test",
        },
    },
    {   
        "metainfo": {
            "name": "Aquarium",
            "classes": ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray')
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/Aquarium",
        "ann_file": "Aquarium Combined.v2-raw-1024.coco/test/annotations_without_background.json",
        "data_prefix": {
            "img": "Aquarium Combined.v2-raw-1024.coco/test"
        }
    },
    {   
        "metainfo": {
            "name": "CottontailRabbits",
            "classes": ("Cottontail-Rabbit",)
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/CottontailRabbits",
        "ann_file": "test/annotations_without_background.json",
        "data_prefix": {
            "img": "test"
        }
    },
    {   
        "metainfo": {
            "name": "EgoHands",
            "classes": ("hand",)
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/EgoHands",
        "ann_file": "generic/test/annotations_without_background.json",
        "data_prefix": {
            "img": "generic/test"
        }
    },
    {   
        "metainfo": {
            "name": "NorthAmericaMushrooms",
            "classes": ("CoW", "chanterelle")
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/NorthAmericaMushrooms",
        "ann_file": "North American Mushrooms.v1-416x416.coco/test/annotations_without_background.json",
        "data_prefix": {
            "img": "North American Mushrooms.v1-416x416.coco/test"
        }
    },
    {   
        "metainfo": {
            "name": "Packages",
            "classes": ("package",)
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/Packages",
        "ann_file": "Raw/test/annotations_without_background.json",
        "data_prefix": {
            "img": "Raw/test"
        }
    },
    {   
        "metainfo": {
            "name": "PascalVOC",
            "classes": ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/PascalVOC",
        "ann_file": "valid/annotations_without_background.json",
        "data_prefix": {
            "img": "valid"
        }
    },
    {   
        "metainfo": {
            "name": "pistols",
            "classes": ("pistol",)
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/pistols",
        "ann_file": "export/test_annotations_without_background.json",
        "data_prefix": {
            "img": "export"
        }
    },
    {   
        "metainfo": {
            "name": "pothole",
            "classes": ("pothole",)
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/pothole",
        "ann_file": "test/annotations_without_background.json",
        "data_prefix": {
            "img": "test"
        }
    },
    {   
        "metainfo": {
            "name": "Raccoon",
            "classes": ("raccoon",)
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/Raccoon",
        "ann_file": "Raccoon.v2-raw.coco/test/annotations_without_background.json",
        "data_prefix": {
            "img": "Raccoon.v2-raw.coco/test"
        }
    },
    {   
        "metainfo": {
            "name": "ShellfishOpenImages",
            "classes": ("Crab", "Lobster", "Shrimp")
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/ShellfishOpenImages",
        "ann_file": "raw/test/annotations_without_background.json",
        "data_prefix": {
            "img": "raw/test"
        }
    },
    {   
        "metainfo": {
            "name": "thermalDogsAndPeople",
            "classes": ("dog", "person")
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/thermalDogsAndPeople",
        "ann_file": "test/annotations_without_background.json",
        "data_prefix": {
            "img": "test"
        }
    },
    {   
        "metainfo": {
            "name": "VehiclesOpenImages",
            "classes": ('Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')
        },
        "data_root": "/mnt/bn/ecomckptnas/yufeizhan/valleyomni_training_yamls/TRY/odinw/VehiclesOpenImages",
        "ann_file": "416x416/test/annotations_without_background.json",
        "data_prefix": {
            "img": "416x416/test"
        }
    }
)
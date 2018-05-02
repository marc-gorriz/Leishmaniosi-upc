# Experiment config file
experiment_name = "exp1"

train_data_path = "[project_path]/data/patches"
test_data_path = "[project_path]/data/balance/test"
test_labels_path = "[project_path]/data/balance/labels"

experiments_path = "[project_path]/experiments"

# Train parameters
tB_write_graph = True
tB_write_images = False

write_test_images = True

input_weights = None

num_classes = 8
batch_size = 5

overall_epochs = 200
parasite_epochs = 20

overall_augmentation = False
parasite_augmentation = False

overall_score = 0.3
parasite_score = 0.001

dropout = 0

# Test parameters
save_predictions = True
save_regions = True
print_overall_table = True
print_jaccard_table = True
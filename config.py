class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    
class Config:
    csv = "../Downloads/downloads/"
    model_type = "mlp"
    label = "multiclass"
    feature = "feat4"
    batch_size = 128
    epochs = 10
    rounds = 10
    learning_rate = 3e-4 #1e-3
    min_available_clients = 2
    fraction_fit = 0.1
    early_stop_patience = 3
    early_stop_monitor = "loss"
    early_stop_min_delta = 1e-4
    early_stop_restore_best_weights = True
    data_train_size = 0.8
    data_test_size = 0.2
    output_path = f"results/{feature}/{label}/"
    performance_file = "performance.csv"
    weights_file = "model_weights.npz"
    output_activation = "softmax"
    bsm = "23bsm"
    min_evaluate_clients = 2
    
# class Config:
#    bsm = "23bsm"
#    csv = f"./VeReMi.csv"
#    fedcsv = f"./VeReMi.csv"
#    model_type = "mlp"
#    label = "atk_2"
#    feature = "feat4"
#    batch_size = 200
#    epochs = 5
#    rounds = 350
#    learning_rate = 1e-3
#    min_available_clients = 2
#    min_evaluate_clients = 2
#    fraction_fit = 1
#    output_activation = "softmax"
#    early_stop_patience = 3
#    early_stop_monitor = "loss"
#    early_stop_min_delta = 1e-4
#    early_stop_restore_best_weights = True
#    data_train_size = 0.8
#    data_test_size = 0.2
#    performance_file = "performance.csv"
#    weights_file = "model_weights.npz"

def check_p_reward_train(data_path):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    
    num_trials = len(data) - 1
    num_correct_train_trials = 0
    num_rewarded_correct_train_trials = 0
    for i in range(1, num_trials + 1):
        curr_trial_info = data[i]
        if curr_trial_info["trial_type"] == "train":
            if curr_trial_info["action"] == curr_trial_info["correct_choice"]:
                num_correct_train_trials = num_correct_train_trials + 1
                if curr_trial_info["reward"] == 1:
                    num_rewarded_correct_train_trials = num_rewarded_correct_train_trials + 1
    print(num_rewarded_correct_train_trials/num_correct_train_trials)

for path in data_paths_across_models[model_names[0]]:
    check_p_reward_train(path)

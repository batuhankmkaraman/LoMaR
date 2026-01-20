def get_example_config():  # without density, ver_1 data, visit aug off, no risk, with long early stop, survival.
    config = {}
    config['exp_id'] = 0
    # model
    config['n_past_visits'] = 5 # present point + 4 years of history
    config['n_future_dx'] = 5 # 5 follow-up years
    config['input_embedding_dim'] = 512
    config['model_embedding_dim'] = 128
    config['n_heads'] = 4
    config['global_do_rate'] = 0.1
    # config['model_weight_dir'] = "/midtier/sablab/scratch/bkk4001/miccai_breast/exp_34/model_weights/rt_1_rv_0_ri_0/i_param_7/model_weights.pth"
    config['model_weight_dir'] = ""
    # data
    config['path_to_csv'] = "demo/data/demo_metadata.csv"
    config['n_pseudo'] = 1 # number of pseudo test sets for evaluation
    # results
    config['results_dir'] = "demo/results/"
    return config
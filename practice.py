import multiprocessing
import math

def calculate_workers_and_rounds(environments, episodes_per_env):
    num_cores = multiprocessing.cpu_count()

    if episodes_per_env == 1:
        num_worker_per_env = 1
    elif episodes_per_env >= 2:
        num_worker_per_env = episodes_per_env // 2
    
    # Calculate total number of workers
    total_num_workers = num_worker_per_env * len(environments)

    if total_num_workers > num_cores:
        rounds = math.ceil(total_num_workers / num_cores) 

        num_worker_per_round = []
        workers_remaining = total_num_workers
        for i in range(rounds):
            if workers_remaining >= num_cores:
                num_worker_per_round.append(num_cores)
                workers_remaining -= num_cores
            else:
                num_worker_per_round.append(workers_remaining)
                workers_remaining = 0
        num_env_per_round = [int(x / num_worker_per_env) for x in num_worker_per_round] #num_worker_per_round / num_worker_per_env
    else:
        rounds = 1
        num_worker_per_round = [total_num_workers]
        num_env_per_round = [len(environments)]
    
    episodes_per_worker = int(episodes_per_env * len(environments) / total_num_workers)
    return num_worker_per_round, num_env_per_round, episodes_per_worker, rounds

def worker_task(env, episodes):
    # Simulate processing the episodes for the environment
    print(f"Processing {episodes} episodes for {env}")

# Example usage
environments = ['env1']  # List of environments
#environments = ['env1', 'env2', 'env3', 'env4', 'env5', 'env6', 'env7', 'env8', 'env9', 'env10', 'env11', 'env12', 'env13', 'env14', 'env15']  # List of environments
episodes_per_env = 12  # Total episodes to sample for each environment

num_workers_per_round, num_env_per_round, episodes_per_worker, rounds = calculate_workers_and_rounds(environments, episodes_per_env)
print(f"Number of workers per round: {num_workers_per_round}")
print(f"Episodes per round: {num_env_per_round}")
print(f"Total rounds needed: {rounds}")

if __name__ == "__main__":
    start_idx = 0   
    for round_number in range(rounds):
        print(f"Starting round {round_number + 1}/{rounds}")
        processes = []
        
        print(f'indices: {start_idx}<->{start_idx+num_env_per_round[round_number]}')
        envs = environments[start_idx:start_idx+num_env_per_round[round_number]]
        for env in envs:
            workers_for_env = num_workers_per_round[round_number] // len(envs)
            for _ in range(workers_for_env):
                p = multiprocessing.Process(target=worker_task, args=(env, episodes_per_worker))
                processes.append(p)
                p.start()
        
        for p in processes:
            p.join()
        start_idx += num_env_per_round[round_number]

#environments = ['env1', 'env2', 'env3', 'env4', 'env5', 'env6', 'env7', 'env8', 'env9', 'env10', 'env11', 'env12', 'env13', 'env14', 'env15']  # List of environments
#environments = ['env1']  # Single environment
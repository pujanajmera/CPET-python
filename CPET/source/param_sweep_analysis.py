import pandas as pd
import matplotlib.pyplot as plt
import argparse

def gen_param_dist_mat(matrix_file, topo_file_list):
    '''
    Takes the matrix file and topo file list for a parameter sweep and 
    returns a well-formatted pandas distance matrix. For replicate runs 
    (ideally 3x), the matrix will be averaged across replicates.

    Inputs:
    matrix_file: Path to the distance matrix file
    topo_file_list: Path to the topo file list; topo file basenames need to be formatted as
    "name_samples_stepsize_replicate.top", and step_size must be every number after
    the decimal place. The name must be non-numerical string, and constant among all files

    Outputs:
    averaged_distances: Averaged distance matrix in pandas format
    '''

    # Load the distance matrix
    distances = pd.read_csv(matrix_file, sep=' ', header=None)
    distances = distances.drop(columns=21)

    # Load the labels
    with open(topo_file_list, 'r') as f:
        labels = [line.strip() for line in f]

    name=labels[0].split("/")[-1].split('_')[0]
    # Modify labels
    labels = [label.replace("_0.top", "").split("/")[-1].replace(name, "") for label in labels]

    # Map each label to its group
    group_map = {label: label.rsplit('_', 1)[0] for label in labels}
    grouped_labels = [group_map[label] for label in labels]

    # Apply the new labels to the DataFrame
    distances.columns = grouped_labels
    distances.index = grouped_labels

    # Aggregate by taking the mean within each group for both rows and columns
    grouped = distances.groupby(level=0).mean()
    averaged_distances = grouped.T.groupby(level=0).mean()

    # Ensure the matrix is symmetric
    averaged_distances = (averaged_distances + averaged_distances.T) / 2

    '''# (Optional) Plot the distance matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(averaged_distances, cmap="Greens_r", linewidths=0.1)
    plt.title("Averaged Distance Matrix")
    plt.show()
    '''
    return averaged_distances

def analyze_param_dist_mat(averaged_distances, threshold=0.1, mode="optimal_field"):
    '''
    Takes in averaged distance matrix and returns optimal list of parameters

    Inputs:
    distances: Averaged distance matrix in pandas format
    threshold: Threshold for the distance matrix to determine either self-consistency
    or converged field
    mode: Mode of analysis; either "self_consistency" or "optimal_field"

    Outputs:
    None (prints optimal parameters)
    '''

    #Generate list of step sizes and samples sweeped over
    sample_list = []
    stepsize_list = []
    for i in reversed(list(averaged_distances.columns)):
        samples = int(i.split("_")[0])
        stepsize = float('0.'+i.split("_")[1])
        if samples not in sample_list:
            sample_list.append(int(samples))
        if stepsize not in stepsize_list:
            stepsize_list.append(stepsize)

    i=0
    updated_dist = 1
    while updated_dist > threshold:
        current_samples=sample_list[i]
        if mode=="optimal_field":
            current_stepsize=stepsize_list[0]
            if i<len(sample_list)-1:
                #First determine if just increasing the number of samples while keeping stepsize at initial value achieves threshold
                updated_dist = averaged_distances.loc[str(sample_list[i])+"_"+str(stepsize_list[0])[2:],str(sample_list[i+1])+"_"+str(stepsize_list[0])[2:]]
                if updated_dist < threshold:
                    print(f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}")
                    return
            #Now, go through list of stepsizes and see off-diagonal distances for threshold
            for j in range(len(stepsize_list)-1):
                updated_dist = averaged_distances.loc[str(sample_list[i])+"_"+str(stepsize_list[j])[2:],str(sample_list[i])+"_"+str(stepsize_list[j+1])[2:]]
                if updated_dist < threshold:
                    current_samples=sample_list[i]
                    current_stepsize=stepsize_list[j]
                    print(f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}")
                    return
                if i==len(sample_list)-1 and j==len(stepsize_list)-2:
                    current_samples=sample_list[i]
                    current_stepsize=stepsize_list[j+1]
                    print(f"Hit sample and stepsize limit, reporting highest sampling values. Next highest sampling values are convergent at distance {updated_dist}")
                    print(f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}")
                    return
        if mode=="self_consistency":
            #Now, go through list of stepsizes and see diagonal distances for threshold
            for j in range(len(stepsize_list)):
                #Compare with the same i and updated j
                updated_dist = averaged_distances.loc[str(sample_list[i])+"_"+str(stepsize_list[j])[2:],str(sample_list[i])+"_"+str(stepsize_list[j])[2:]]
                if updated_dist < threshold:
                    current_stepsize=stepsize_list[j]
                    print(f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}")
                    return
                if i==len(sample_list)-1 and j==len(stepsize_list)-1:
                    current_stepsize=stepsize_list[j]
                    print(f"Hit sample and stepsize limit, reporting highest sampling values. Final distance is at {updated_dist}")
                    print(f"Best param for conver={threshold}: \nSample number: {current_samples} \nStepsize: {current_stepsize}")
                    return
        i+=1

def main():
    parser=argparse.ArgumentParser(description="Analyze parameter sweep distance matrix")
    parser.add_argument("matrix_file", help="Path to the distance matrix file")
    parser.add_argument("topo_file_list", help="Path to the topo file list")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for the distance matrix to determine either self-consistency or converged field")
    parser.add_argument("--mode", type=str, default="optimal_field", help="Mode of analysis; either 'self_consistency' or 'optimal_field'")
    args=parser.parse_args()
    averaged_distances=gen_param_dist_mat(args.matrix_file, args.topo_file_list)
    analyze_param_dist_mat(averaged_distances, args.threshold, args.mode)

if __name__=="__main__":
    main()

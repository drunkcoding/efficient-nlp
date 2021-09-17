import pandas as pd
import os
import subprocess

task_names = ['CoLA', 'MNLI-m', 'MNLI-mm', 'MRPC', 'SST-2', 'STS-B', 'QQP', 'QNLI', 'RTE', 'WNLI']

tsv = pd.DataFrame({"id": [x for x in range(1104)], "label": ["1"]*1104})
tsv.to_csv(f"TestResult/AX.tsv", sep='\t',index=False)

for task in task_names:
    print(task)
    data_dir = f"/data/GlueData/{task}"
    data_file = "test.tsv"
    if task == 'MNLI-m':
        data_dir = data_dir.split('-')[0]
        data_file = "test_matched.tsv"
        # tsv = pd.read_csv(os.path.join(data_dir, "test_matched.tsv"), sep='\t', index_col=False, header=0, error_bad_lines=False)
    elif task == 'MNLI-mm':
        data_dir = data_dir.split('-')[0]
        data_file = "test_mismatched.tsv"
        # tsv = pd.read_csv(os.path.join(data_dir, "test_mismatched.tsv"), sep='\t', index_col=False, header=0, error_bad_lines=False)
    # else:
        # tsv = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', index_col=False, header=0, error_bad_lines=False)
    
    result = subprocess.run(['wc', '-l', os.path.join(data_dir, data_file)], stdout=subprocess.PIPE)
    result = result.stdout
    num_lines = int(result.split()[0])
    ids = [x for x in range(num_lines)]
    labels = ["entailment"]*num_lines if task in ['RTE', 'QNLI', 'MNLI-m', 'MNLI-mm'] else ["1"]*num_lines
    # tsv = tsv[["index"]]
    # tsv = tsv.rename(columns={'index': 'id'})
    # tsv['label'] = "entailment" if task in ['RTE', 'QNLI', 'MNLI-m', 'MNLI-mm'] else "0"
    tsv = pd.DataFrame({"id": ids, "label": labels})
    tsv.to_csv(f"TestResult/{task}.tsv", sep='\t',index=False)
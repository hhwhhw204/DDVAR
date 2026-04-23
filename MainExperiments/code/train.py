import os
import time
import sys
import random
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import logging
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from Bio import SeqIO
# from Bio.PDB.DSSP import DSSP
import pickle
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from model.model_diff import Model as Model



def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  # Set the random seed for numpy
    torch.manual_seed(1)  # seed for PyTorch CPU
    torch.cuda.manual_seed(1)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(1)  # seed for all PyTorch GPUs


def create_dir(path):
    if not os.path.exists(os.path.dirname(path)): 
        os.makedirs(os.path.dirname(path))

 
class FocalLoss(nn.Module):
	def __init__(self,alpha=0.25,gamma=2):
		super(FocalLoss,self).__init__()
		self.alpha=alpha
		self.gamma=gamma
	
	def forward(self,preds,labels):
		eps=1e-7
		loss_1=-1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
		loss_0=-1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
		loss=loss_0+loss_1
		return torch.mean(loss)
     


def rightness(predictions, labels):
    predictions = predictions.cpu()
    labels = labels.cpu()
    preds = (predictions > 0.5).float()
    acc = accuracy_score(labels, preds)
    return acc



def load_npy(file_dict, path_prefix, file_suffix, dataset, row):
    file_name = f"{dataset}_{row['chr']}:{row['start']}:{row['end']}:{row['ref']}:{row['alt']}_{file_suffix}.npy"
    if file_name in file_dict:
        return np.expand_dims(np.load(os.path.join(path_prefix, file_dict[file_name])), axis=0)
    else:
        raise FileNotFoundError(f"{file_name} not found in {path_prefix}")
    

def stack_esm_npy(df, dataset, repr_subfolder, contact_subfolder, seqtype):
    repr_files = {f: f for f in os.listdir(repr_subfolder) if f.endswith('.npy')}
    contact_files = {f: f for f in os.listdir(contact_subfolder) if f.endswith('.npy')}
    reprs = np.vstack([
        load_npy(repr_files, repr_subfolder, seqtype, dataset, row)
        for _, row in df.iterrows()
    ])
    contacts = np.vstack([
        load_npy(contact_files, contact_subfolder, f"{seqtype}_contact", dataset, row)
        for _, row in df.iterrows()
    ])
    reprs = torch.from_numpy(reprs)
    contacts = torch.from_numpy(contacts)
    return reprs, contacts



def dssp_mapping(dssp):
    SS_type = "HBEGITS-"
    dssp_tensor = torch.tensor([SS_type.find(ss) + 1 for ss in dssp])
    fixed_length = 200
    padding_length = max(0, fixed_length - dssp_tensor.size(0))
    dssp_padded = torch.cat([dssp_tensor, torch.full((padding_length,), 0)], dim=0)[:fixed_length]
    return dssp_padded


def asseq_mapping(asseq):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    asseq_tensor = torch.tensor([amino_acids.find(ss) + 1 for ss in asseq])
    fixed_length = 200
    padding_length = max(0, fixed_length - asseq_tensor.size(0))
    asseq_padded = torch.cat([asseq_tensor, torch.full((padding_length,), 0)], dim=0)[:fixed_length]
    return asseq_padded


def process_graph(reprs, contacts, dssp, asseq, pssm, args):
    graph = []
    print("reprs.shape[0]=", reprs.shape[0])
    for i in range(reprs.shape[0]):
        # Create a 200x200 matrix and fill it with values greater than the threshold
        padded_contacts = torch.full((200, 200), args.threshold + 1)
        # Find the first index where the entire column is 0. If not found, return 200.
        indice = torch.where(torch.sum(contacts[i], dim=0) == 0)[0][0].item() if torch.any(torch.sum(contacts[i], dim=0) == 0) else contacts[i].shape[0]
        # Obtain the valid submatrix of the contacts
        sub_contacts = contacts[i][:indice, :indice]
        # Fill in the effective value
        padded_contacts[:indice, :indice] = sub_contacts 
        # Values less than threshold are marked as 1, otherwise they are marked as 0.
        contact = torch.where(padded_contacts <= args.threshold, torch.tensor(1), torch.tensor(0))
        # Set the diagonal line to 1
        indices = torch.arange(200) 
        contact[indices, indices] = 1     
        # Sparse matrix
        edge_index, edge_weight = dense_to_sparse(contact)
        # Convert letters to numbers
        dssp_pad = dssp_mapping(dssp[i])
        asseq_pad = asseq_mapping(asseq[i])
        # Construct a graph object
        data = Data(esm_repr=reprs[i], edge_index=edge_index, edge_attr=edge_weight, dssp=dssp_pad, asseq=asseq_pad, pssm = pssm[i])
        graph.append(data)
    return graph


def stack_pssm(df, dataset, pssm_subfolder, tag):
    all_pssm_data = []
    for index, row in df.iterrows():
        file_name = f"{dataset}_{row['chr']}:{row['start']}:{row['end']}:{row['ref']}:{row['alt']}_{tag}.pssm"
        file_path = os.path.join(pssm_subfolder, file_name)
        # Initialize a 200x20 tensor and fill it with 0.
        matrix_tensor = torch.zeros((200, 20+20), dtype=torch.int)  

        if not os.path.exists(file_path):
            print(f"pssm file:{file_path} not exist!")
            sys.exit(0)

        if os.path.exists(file_path):
            pssm_data = pd.read_csv(os.path.join(pssm_subfolder,file_name), delim_whitespace=True, header=None, skiprows=3, skipfooter=5,  engine='python')
            matrix_data = pssm_data.iloc[:, 2:22+20].values
            matrix_tensor[:matrix_data.shape[0], :matrix_data.shape[1]] = torch.tensor(matrix_data, dtype=torch.int)
        all_pssm_data.append(matrix_tensor)
    all_pssm_tensor = torch.stack(all_pssm_data)
    return all_pssm_tensor


def stack_dssp(df, dataset, dssp_subfolder, tag):
    all_dssp_data = []
    for index, row in df.iterrows():
        file_name = f"{dataset}_{row['chr']}:{row['start']}:{row['end']}:{row['ref']}:{row['alt']}_{tag}.fasta"
        file_path = os.path.join(dssp_subfolder, file_name)
        record = next(SeqIO.parse(file_path, "fasta"))
        sequence = str(record.seq) 
        all_dssp_data.append(sequence)
    return all_dssp_data


def stack_asseq(df, dataset, asseq_subfolder, tag):
    all_asseq_data = []
    for index, row in df.iterrows():
        file_name = f"{dataset}_{row['chr']}:{row['start']}:{row['end']}:{row['ref']}:{row['alt']}_{tag}.fasta"
        file_path = os.path.join(asseq_subfolder, file_name)
        record = next(SeqIO.parse(file_path, "fasta"))
        sequence = str(record.seq) 
        all_asseq_data.append(sequence)
    return all_asseq_data



def monopathy_train_test_split():
    disease_test_dfs = []   # Store the test set for all diseases
    diseases = ['epilepsy', 'Phenylketonuria', 'Malignant_hyperthermia', 'Cystic_Fibrosis']
    for disease in diseases:
        csv_path = f"../../../../Preprocessing/Disgenet/out/Disgenet_{disease}.tsv"
        disease_df = pd.read_csv(csv_path, sep = '\t')
        disease_train, disease_test = train_test_split(
            disease_df, 
            test_size=0.8, 
            random_state=42,
        )
        disease_test_dfs.append(disease_test)

        save_path = f"../out/monopathy_test_dataset/{disease}_test.tsv"
        if not os.path.exists(save_path):
            disease_test.to_csv(save_path, sep='\t', index=False)
    
    all_disease_test_variants = pd.concat(disease_test_dfs, ignore_index=True)
    all_disease_test_variants = all_disease_test_variants.drop_duplicates(subset=['chr', 'start', 'end', 'ref', 'alt'])
    all_disease_test_variants.to_csv(f"../out/monopathy_test_dataset/{len(diseases)}_disease_test.tsv", sep='\t', index=False)


def exclude_monopathy(df):
    all_disease_test = pd.read_csv("../out/monopathy_test_dataset/4_disease_test.tsv", sep = '\t')
    # Exclude the test set data
    disease_key = all_disease_test[['chr', 'start', 'end', 'ref', 'alt']].astype(str).agg(':'.join, axis=1)
    df_key = df[['chr', 'start', 'end', 'ref', 'alt']].astype(str).agg(':'.join, axis=1)
    mask = ~df_key.isin(disease_key)
    df_filtered = df[mask].copy()
    print(f"Filtered out {len(df) - len(df_filtered)} variants, remaining {len(df_filtered)} variants")
    return df_filtered


def obtain_positive_data(args):
    csv_path = f"{args.relative_path}/csv/{args.pos_dataset}/{args.pos_dataset}"
    repr_path = f"{args.relative_path}/prt_repr/{args.pos_dataset}/"
    contact_path = f"{args.relative_path}/contact/{args.pos_dataset}/"
    dssp_path = f"{args.relative_path}/dssp/{args.pos_dataset}"
    asseq_path = f"{args.relative_path}/prt_asseq/{args.pos_dataset}/"
    pssm_path = f"{args.relative_path}/pssm/{args.pos_dataset}/"

    df = pd.read_csv(f"{csv_path}_wz_alterpos.tsv",sep="\t")
    # df = exclude_monopathy(df)
    ref_reprs, ref_contacts = stack_esm_npy(df, f"{args.pos_dataset}", f"{repr_path}/ref/", f"{contact_path}/ref/", "ref")
    alt_reprs, alt_contacts = stack_esm_npy(df, f"{args.pos_dataset}", f"{repr_path}/alt/", f"{contact_path}/alt/", "alt")
    
    ref_asseq = stack_asseq(df, f"{args.pos_dataset}", f"{asseq_path}/ref/", "ref")
    alt_asseq = stack_asseq(df, f"{args.pos_dataset}", f"{asseq_path}/alt/", "alt")

    ref_pssm = stack_pssm(df, f"{args.pos_dataset}", f"{pssm_path}/ref/","ref")
    alt_pssm = stack_pssm(df, f"{args.pos_dataset}", f"{pssm_path}/alt/","alt")

    ref_dssp = stack_dssp(df, f"{args.pos_dataset}", f"{dssp_path}/ref/","ref")
    alt_dssp = stack_dssp(df, f"{args.pos_dataset}", f"{dssp_path}/alt/","alt")

    ref_graphs = process_graph(ref_reprs, ref_contacts, ref_dssp, ref_asseq, ref_pssm, args)
    alt_graphs = process_graph(alt_reprs, alt_contacts, alt_dssp, alt_asseq, alt_pssm, args)

    # torch.save(ref_graphs, f"../tmp/diff_positive_ref_graphs_thre_{args.threshold}.pt")
    # torch.save(alt_graphs, f"../tmp/diff_positive_alt_graphs_thre_{args.threshold}.pt")

    df = df[['chr','start','end','ref','alt']]
    return df, ref_graphs, alt_graphs



def obtain_negative_data(args):
    pos_csv_path = f"{args.relative_path}/csv/{args.pos_dataset}/{args.pos_dataset}_wz_alterpos.tsv"
    pos_df = pd.read_csv(pos_csv_path,sep="\t",usecols = ['chr', 'start', 'end', 'ref', 'alt'])
    pos_df['chr'] = pos_df['chr'].astype(str)

    neg_csv_path = f"{args.relative_path}/csv/{args.neg_dataset}/{args.neg_dataset}_wz_alterpos.tsv" 
    neg_df = pd.read_csv(neg_csv_path,sep="\t")
    neg_df['chr'] = neg_df['chr'].astype(str)

    if args.pos_dataset != args.neg_dataset:
        # Exclude the positive sample data from the negative samples.
        merged_df = neg_df.merge(pos_df, on=['chr', 'start', 'end', 'ref', 'alt'], how='left', indicator=True)
        neg_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        neg_sampled = neg_df.sample(n=pos_df.shape[0]*args.mul, random_state=args.seed)
        # neg_sampled.to_csv(f"../tmp/neg_sample_{args.mul}.tsv",index=False)
    else:
        neg_sampled = neg_df

    repr_path = f"{args.relative_path}/prt_repr/{args.neg_dataset}/"
    contact_path = f"{args.relative_path}/contact/{args.neg_dataset}/"
    dssp_path = f"{args.relative_path}/dssp/{args.neg_dataset}"
    asseq_path = f"{args.relative_path}/prt_asseq/{args.neg_dataset}/"
    pssm_path = f"{args.relative_path}/pssm/{args.neg_dataset}/"

    ref_reprs, ref_contacts = stack_esm_npy(neg_sampled, args.neg_dataset, f"{repr_path}/ref/", f"{contact_path}/ref/", "ref")
    alt_reprs, alt_contacts = stack_esm_npy(neg_sampled, args.neg_dataset, f"{repr_path}/alt/", f"{contact_path}/alt/", "alt")

    ref_dssp = stack_dssp(neg_sampled, f"{args.neg_dataset}", f"{dssp_path}/ref/","ref")
    alt_dssp = stack_dssp(neg_sampled, f"{args.neg_dataset}", f"{dssp_path}/alt/","alt")

    ref_asseq = stack_asseq(neg_sampled, args.neg_dataset, f"{asseq_path}/ref/", "ref")
    alt_asseq = stack_asseq(neg_sampled, args.neg_dataset, f"{asseq_path}/alt/", "alt")

    ref_pssm = stack_pssm(neg_sampled, f"{args.neg_dataset}", f"{pssm_path}/ref/","ref")
    alt_pssm = stack_pssm(neg_sampled, f"{args.neg_dataset}", f"{pssm_path}/alt/","alt")

    ref_graphs = process_graph(ref_reprs, ref_contacts, ref_dssp, ref_asseq, ref_pssm, args)
    alt_graphs = process_graph(alt_reprs, alt_contacts, alt_dssp, alt_asseq, alt_pssm, args)

    # torch.save(ref_graphs, f"../tmp/diff_negative_ref_graphs_mul_{args.mul}_thre_{args.threshold}.pt")
    # torch.save(alt_graphs, f"../tmp/diff_negative_alt_graphs_mul_{args.mul}_thre_{args.threshold}.pt")
    
    neg_sampled = neg_sampled[['chr','start','end','ref','alt']]
    return neg_sampled, ref_graphs, alt_graphs


def k_verify(RefGraphs, AltGraphs, Y, k, args):
    n = len(RefGraphs)
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]

    aurocs, auprcs, f1_scores = [], [], []
    x1,x2 = [],[]
    min_val_loss = 100
    max_val_auroc = 0
    
    if args.mode == 'train':
        # Obtain the current timestamp and format it as a folder name
        timestamp = time.strftime("%Y%m%d-%H%M", time.localtime())
        output_dir = os.path.join("../out/save_model", f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    elif args.mode == 'reproduce':
        output_dir = args.model_dir
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Model directory {output_dir} not found!")
        print(f"Reproduce mode: loading models from {output_dir}")

    for i in range(1, k + 1):
        # Divide the dataset
        ix = assignments == i
        ref_graph_train, ref_graph_test = [g for j, g in enumerate(RefGraphs) if not ix[j]], [g for j, g in enumerate(RefGraphs) if ix[j]]
        alt_graph_train, alt_graph_test = [g for j, g in enumerate(AltGraphs) if not ix[j]], [g for j, g in enumerate(AltGraphs) if ix[j]]
        y_train, y_test = Y[~ix], Y[ix]

        # Build a PyG graph dataset
        train_dataset = list(zip(ref_graph_train, alt_graph_train, y_train))
        test_dataset = list(zip(ref_graph_test, alt_graph_test, y_test))

        train_loader = GeoDataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = GeoDataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=0, pin_memory=True)

        # Initialization model, loss function and optimizer
        net = Model().cuda()
        if args.loss == 'BCEloss':
            criterion = nn.BCELoss()
        elif args.loss == 'Focalloss':
            criterion = FocalLoss(alpha=0.5, gamma=2)
        else:
            print("no loss")
            sys.exit(0)
            
        optimizer = optim.Adam(params=net.parameters(), lr=args.lr, betas=(0.9, 0.999),weight_decay=1e-5)

        if args.mode == 'reproduce':
            # In the reproduction mode, directly load the corresponding model.
            model_files = [f for f in os.listdir(output_dir) if f.startswith(f"model_fold_{i}_") and f.endswith('.pth')]
            if not model_files:
                raise FileNotFoundError(f"No model file found for fold {i} in {output_dir}")

            model_files.sort(reverse=True)
            model_path = os.path.join(output_dir, model_files[0])
            net.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
            
            # Make a direct prediction
            net.eval()
            y_true_total, y_pred_total = torch.tensor([]), torch.tensor([])
            for ref_graph, alt_graph, y_true in test_loader:
                ref_graph, alt_graph, y_true = [x.cuda() for x in (ref_graph, alt_graph, y_true)]
                with torch.no_grad():
                    y_pred,_ = net(ref_graph, alt_graph)
                    y_true_total = torch.cat((y_true_total, y_true.cpu()))
                    y_pred_total = torch.cat((y_pred_total, y_pred.cpu()))
            
            auroc = roc_auc_score(y_true_total, y_pred_total)
            auprc = average_precision_score(y_true_total, y_pred_total)
            aurocs.append(auroc)
            auprcs.append(auprc)

            y_pred_binary = np.where(np.array(y_pred_total) >= 0.5, 1, 0)
            f1 = f1_score(y_true_total, y_pred_binary)
            f1_scores.append(f1)

            for k_val in y_pred_total[y_true_total == 0]:
                x1.append(k_val)
            for j in y_pred_total[y_true_total == 1]:
                x2.append(j)
            
            val_msg = f'Fold {i} (Reproduce): auroc={auroc:.4f}, auprc={auprc:.4f}, f1_score={f1:.4f}'
            print(val_msg)
            logging.info(val_msg)
            continue


        y_true_total, y_pred_total = torch.tensor([]), torch.tensor([])
        val_losses = []
        for epoch in tqdm(range(args.e)):
            net.train()
            for batch_idx, (ref_graph, alt_graph, y_true) in enumerate(train_loader):
                ref_graph, alt_graph, y_true = [x.cuda() for x in (ref_graph, alt_graph, y_true)]
                y_pred, kl_loss = net(ref_graph, alt_graph)
                loss = criterion(y_pred, y_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                right = rightness(y_pred, y_true)
                if epoch % 1  == 0 and batch_idx == 0:
                    msg = '{} part  Training epoch:{}  Training Loss: {:.6f}  Training accuracy: {:.2f}%'.format(
                        i,  epoch, batch_idx *
                        args.bs, len(train_loader.dataset), 100.0 *
                        batch_idx / len(train_loader), loss.data,
                        100. * right)
                    print(msg)
                    logging.info(msg)

        net.eval()
        for ref_graph, alt_graph, y_true in test_loader:
            ref_graph, alt_graph, y_true = [x.cuda() for x in (ref_graph, alt_graph, y_true)]
            with torch.no_grad():
                y_pred,_ = net(ref_graph, alt_graph)
                val_loss = criterion(y_pred, y_true)
                y_true_total = torch.cat((y_true_total, y_true.cpu()))
                y_pred_total = torch.cat((y_pred_total, y_pred.cpu()))
                val_losses.append(val_loss)      

        auroc  = roc_auc_score(y_true_total, y_pred_total)
        auprc = average_precision_score(y_true_total, y_pred_total)
        aurocs.append(auroc)
        auprcs.append(auprc)

        y_pred_binary = np.where(np.array(y_pred_total) >= 0.5, 1, 0)
        f1 = f1_score(y_true_total, y_pred_binary)
        f1_scores.append(f1)

        for k in y_pred_total[y_true_total == 0]:
            x1.append(k)
        for j in y_pred_total[y_true_total == 1]:
            x2.append(j)
        
        val_msg = 'Validation set: auroc={:.4f}, auprc={:.4f}, f1_score={:.4f}'.format(auroc,auprc,f1)
        print(val_msg)
        logging.info(val_msg)

        if sum(val_losses) < min_val_loss:
            print("best model update!")
            logging.info("best model update!")
            min_val_loss = sum(val_losses)
            
        model_filename = os.path.join(output_dir, f"model_fold_{i}_mul_{args.mul}_auroc_{auroc:.4f}.pth")
        torch.save(net.state_dict(), model_filename)
        print('\n')

    mean_auroc = np.mean(aurocs, axis=0)
    mean_auprc = np.mean(auprcs, axis=0)
    mean_f1_score = np.mean(f1_scores, axis=0)
    statistic, pvalue = stats.mannwhitneyu(x1, x2, use_continuity=True, alternative='two-sided')

    logging.info(str(net))
    result_msg = f"K fold Result:\nMean AUROC={mean_auroc:.4f}\nMean AUPRC={mean_auprc:.4f}\nMean F1={mean_f1_score:.4e}"
    print(result_msg)
    logging.info(result_msg)
    return mean_auroc, mean_auprc, mean_f1_score



def write_args_to_txt(args, file_path):
    with open(file_path, 'a') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\t")
        f.write(f"{arg}: {value}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--mode', type=str, default='reproduce', choices=['train', 'reproduce'],
                        help='train: train new model, reproduce: load existing model for prediction')
    parser.add_argument('--model_dir', type=str, default="../out/save_model/run_20260330-1020/",
                        help='path to saved model directory (used when mode=reproduce)')
    parser.add_argument("--pos_dataset", default="Testset") 
    parser.add_argument("--neg_dataset", default="Testset")
    parser.add_argument('--mul', type=int, default=1)
    parser.add_argument('--e', type=int, default=12)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='BCEloss')
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--relative_path', default="../../ObtainFeature/out")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed) 

    # monopathy_train_test_split()
    
    log_file = os.path.join("../out/log/", f"{args.mode}_{args.pos_dataset}_{args.neg_dataset}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    write_args_to_txt(args,log_file)

    if args.mode == 'reproduce':
        print("using exist data...")  
        pos_ref_graphs = torch.load(f"../tmp/diff_positive_ref_graphs_thre_{args.threshold}.pt")
        pos_alt_graphs = torch.load(f"../tmp/diff_positive_alt_graphs_thre_{args.threshold}.pt")
        neg_ref_graphs = torch.load(f"../tmp/diff_negative_ref_graphs_mul_{args.mul}_thre_{args.threshold}.pt")
        neg_alt_graphs = torch.load(f"../tmp/diff_negative_alt_graphs_mul_{args.mul}_thre_{args.threshold}.pt")
    else:
        print("data loading...")  
        pos_df, pos_ref_graphs, pos_alt_graphs = obtain_positive_data(args)
        print("positive data load finish...") 
        neg_df, neg_ref_graphs, neg_alt_graphs = obtain_negative_data(args)
        print("negative data load finish...")

    # Merging positive and negative samples
    ref_graphs = pos_ref_graphs + neg_ref_graphs
    alt_graphs = pos_alt_graphs + neg_alt_graphs
    labels = torch.tensor([1] * len(pos_ref_graphs) + [0] * len(neg_ref_graphs), dtype=torch.float32)

    auroc, auprc, f1_score = k_verify(ref_graphs, alt_graphs, labels, 5, args)
    log_message = f"{str(auroc)[:6]}\t{str(auprc)[:6]}\t{str(f1_score)[:6]}"
    logging.info(log_message)



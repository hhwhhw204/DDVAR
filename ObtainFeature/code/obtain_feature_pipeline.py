import os
import sys
import io
import re
import argparse
import subprocess
import warnings
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import concurrent.futures

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Bio import PDB, SeqIO
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning

os.environ['TORCH_HOME'] = ''
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', BiopythonWarning)


def create_dir(path):
    if path.endswith(os.path.sep) or not os.path.splitext(path)[1]:
        os.makedirs(path, exist_ok=True) 
    else:
        print(os.path.dirname(path))
        os.makedirs(os.path.dirname(path), exist_ok=True) 


def obtain_prt_cut(asseq, variant_file, args):
    records = list(SeqIO.parse(asseq, "fasta"))
    variant_df = pd.read_csv(variant_file, header=None, sep="\t", engine = 'python').iloc[:,[0,1,3,4,5,6,7,2]]
    variant_df.columns = ['lineid','type','chr','start','end','ref','alt','protein_change']
    variant_df = variant_df[variant_df['type'] == 'nonsynonymous SNV'].drop(columns=['type'])
    variant_df['gene'] = variant_df['protein_change'].apply(
        lambda x: x.split(",")[0].split(":")[0] if isinstance(x, str) else x
    )
    variant_df['protein_change'] = variant_df['protein_change'].apply(
        lambda x: x.split(",")[0].split(":")[-1] if isinstance(x, str) else x
    )

    csv_path = f"../out/csv/{args.folder}/{args.dataset}_wz_alterpos.tsv"
    create_dir(csv_path)
    columns = ['lineid', 'chr', 'start', 'end', 'ref', 'alt', 'alterpos', 'gene']
    if os.path.exists(csv_path):
        os.remove(csv_path)
    pd.DataFrame(columns=columns).to_csv(csv_path, sep='\t', index=False)

    create_dir(f"../out/prt_asseq/{args.folder}/ref/")
    create_dir(f"../out/prt_asseq/{args.folder}/alt/")


    for i in tqdm(range(0, len(records), 2)):
        # [:-1] is used to exclude *
        ref_record = records[i][:-1] 
        alt_record = records[i+1][:-1]
        info = alt_record.description.split(" ")
        type = info[4]
        lineid = info[0]

        matched_variant = variant_df[variant_df['lineid'] == lineid]
        if matched_variant.empty:
            continue

        description = args.dataset + '_' + \
            str(matched_variant.iloc[0]['chr']) + ':' + \
            str(matched_variant.iloc[0]['start']) + ':' + \
            str(matched_variant['end'].iloc[0]) + ':' + \
            matched_variant.iloc[0]['ref'] + ':' + \
            matched_variant.iloc[0]['alt']

        if type == 'protein-altering':
            alter_pos = int(info[7].split("-")[0]) - 1
            start = max(0, alter_pos - args.prt_cut_len)
            end = min(len(ref_record.seq), alter_pos + args.prt_cut_len)

            ref_record = SeqRecord(Seq(ref_record.seq[start:end]), id=ref_record.description, description= description + "_ref")
            alt_record = SeqRecord(Seq(alt_record.seq[start:end]), id=alt_record.description, description= description + "_alt")

            matched_variant.to_csv(csv_path, sep="\t", index=False, header=False, mode='a')
            
            save_ref_asseq = f"../out/prt_asseq/{args.folder}/ref/{description}_ref.fasta"
            with open(save_ref_asseq, "w") as output_handle:
                SeqIO.write(ref_record, output_handle, "fasta")

            save_alt_asseq = f"../out/prt_asseq/{args.folder}/alt/{description}_alt.fasta"
            with open(save_alt_asseq, "w") as output_handle:
                SeqIO.write(alt_record, output_handle, "fasta")




def protein_embedding_piece(records_path, model, batch_converter, seqtype, args):
    records = list(SeqIO.parse(records_path, "fasta"))
    save_dir = f"../out/prt_repr/{args.folder}/{seqtype}/"
    create_dir(save_dir)

    max_tokens_len = args.prt_cut_len *2 + 2
    for record in tqdm(records):
        description = str(record.description).split(" ")[-1]
        save_path = save_dir + f"{description}.npy"

        _, _, tokens = batch_converter([(record.description, str(record.seq)),])
        tokens_len = tokens.shape[1]

        with torch.no_grad():
            results = model(tokens.cuda(), repr_layers=[12], return_contacts=True)
            token_repr = results["representations"][12][:, 1: tokens_len-1, :].cpu()   

        if tokens_len < max_tokens_len: 
            token_repr = F.pad(token_repr, pad=(0, 0, 0, max_tokens_len - tokens_len), mode='constant', value=0)
        np.save(save_path, token_repr[0].numpy())
        


def calculate_distance(residue1, residue2):
    try:
        atom1 = residue1['CA']
        atom2 = residue2['CA']
        distance = atom1 - atom2   # Biopython supports calculating distances directly
        return distance
    except KeyError:
        return np.inf


# Generate contact map from PDB structure
def generate_contact_map(structure):
    residues = [res for res in structure.get_residues() if PDB.is_aa(res, standard=True)]
    num_residues = len(residues)
    contact_map = np.zeros((num_residues, num_residues))
    for i in range(num_residues):
        for j in range(i+1, num_residues):
            distance = calculate_distance(residues[i], residues[j])
            contact_map[i, j] = distance
            contact_map[j, i] = distance 
    return contact_map



def pad_contact_map(contact_map, target_size):
    num_residues = contact_map.shape[0]
    padded_map = np.zeros((target_size, target_size))
    padded_map[:num_residues, :num_residues] = contact_map
    return padded_map


def obtain_mutilmodal_features(args):
    print("loading...")
    esm2, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
    esm2 = esm2.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    print("loading esm2 model finish!")

    import esm
    esmfold = esm.pretrained.esmfold_v1()
    esmfold = esmfold.eval().cuda()
    print("loading esmfold model finish!")

    ref_asseq_folder = f"../out/prt_asseq/{args.folder}/ref"
    alt_asseq_folder = f"../out/prt_asseq/{args.folder}/alt"
    process_fasta(esmfold, esm2, batch_converter, ref_asseq_folder, args, "ref")
    process_fasta(esmfold, esm2, batch_converter, alt_asseq_folder, args, "alt")




def process_fasta(esmfold, esm2, batch_converter, asseq_folder, args, seqtype):
    # Base directories for different datasets
    base_pdb_dir = "../out/pdb"
    base_contact_dir = "../out/contact"
    base_dssp_dir = "../out/dssp"
    base_pssm_dir = "../out/pssm"
    base_repr_dir = "../out/prt_repr"
    
    # Construct directory paths
    pdb_dir = os.path.join(base_pdb_dir, args.folder, seqtype)
    contact_dir = os.path.join(base_contact_dir, args.folder, seqtype)
    dssp_dir = os.path.join(base_dssp_dir, args.folder, seqtype)
    pssm_dir = os.path.join(base_pssm_dir, args.folder, seqtype)
    repr_dir = os.path.join(base_repr_dir, args.folder, seqtype)

    # Create necessary directories
    for dir_path in [pdb_dir, contact_dir, dssp_dir, pssm_dir, repr_dir]:
        create_dir(dir_path)
    
    csv_path = f"../out/csv/{args.folder}/{args.dataset}_wz_alterpos.tsv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.read_csv(csv_path, header=None, sep='\t')
    df.columns = ['line','chr','start','end','ref','alt','protein_change', 'gene']

    df['id'] = (
        args.dataset + '_' +
        df['chr'].astype(str) + ':' + df['start'].astype(str) + ':' + df['end'].astype(str) + ':' +
        df['ref'] + ':' + df['alt'] + '_' + seqtype + '.fasta'
    )

    for asseq_name in tqdm((os.listdir(asseq_folder))):
        asseq_path = os.path.join(asseq_folder, asseq_name)
        # Read fasta file and extract sequence and description
        with open(asseq_path, "r") as handle:
            record = list(SeqIO.parse(handle, "fasta"))[0]
        sequence = str(record.seq)
        description = str(record.description).split(" ")[-1]

        # save_path
        pdb_path = os.path.join(pdb_dir, f"{description}.pdb")
        contact_path = os.path.join(contact_dir, f"{description}_contact.npy")
        dssp_path = os.path.join(dssp_dir, f"{description}.fasta")
        pssm_path = os.path.join(pssm_dir, f"{description}.pssm")
        repr_path = os.path.join(repr_dir, f"{description}.npy")
        
        # Check to ensure that all the files are present.
        if all(os.path.exists(path) for path in [pssm_path, pdb_path, contact_path, dssp_path, repr_path]):
            print(f"All files exist for {description}, continue")
            continue

        # PSSM
        if not os.path.exists(pssm_path):
            for i in range(3):
                run_psiblast(asseq_path, pssm_path, args)
                if os.path.exists(pssm_path):
                    print(f"Processed {description}: PSSM saved.")
                    break
                print(f"Attempt {i+1} failed")
            else:
                print(f"Failed to process {description}: PSSM not generated after 3 attempts.")
                sys.exit(0)

        if not os.path.exists(dssp_path) or not os.path.exists(contact_path):
            # PDB
            if not os.path.exists(pdb_path):
                with torch.no_grad():
                    output = esmfold.infer_pdb(sequence)
                with open(pdb_path, "w") as f:
                    f.write(output)
                print(f"Processed {description}: PDB saved.")
            else:
                with open(pdb_path, "r") as f:
                    output = f.read()

            # contact map
            if not os.path.exists(contact_path):
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure('protein', io.StringIO(output))
                contact_map = generate_contact_map(structure)
                if contact_map.shape[0] < args.prt_cut_len*2:
                    contact_map = pad_contact_map(contact_map, args.prt_cut_len*2)
                np.save(contact_path, contact_map)
                print(f"Processed {description}: contact map saved.")

            # DSSP 
            if not os.path.exists(dssp_path):
                if 'structure' not in locals():
                    parser = PDB.PDBParser(QUIET=True)
                    structure = parser.get_structure('protein', io.StringIO(output))
                dssp = DSSP(structure[0], pdb_path, dssp=args.mkdssp_path)
                secondary_structure = ''.join([res[2] for res in dssp])
                with open(dssp_path, 'w') as fasta_file:
                    fasta_file.write(f">{description}\n")
                    fasta_file.write(f"{secondary_structure}\n")
                print(f"Processed {description}: DSSP saved.")
        
        # repr
        if not os.path.exists(repr_path):
            max_tokens_len = args.prt_cut_len *2 + 2
            description = str(record.description).split(" ")[-1]
            _, _, tokens = batch_converter([(record.description, str(record.seq)),])
            tokens_len = tokens.shape[1]

            with torch.no_grad():
                results = esm2(tokens.cuda(), repr_layers=[12], return_contacts=False)
                token_repr = results["representations"][12][:, 1: tokens_len-1, :].cpu()    
            if tokens_len < max_tokens_len:     
                token_repr = F.pad(token_repr, pad=(0, 0, 0, max_tokens_len - tokens_len), mode='constant', value=0)
            np.save(repr_path, token_repr[0].numpy())
            print(f"Processed {description}: ESM2 repr saved.")

        torch.cuda.empty_cache()




def run_psiblast(fasta_filename, pssm_filename, args):
    cmd = [
        "timeout", "3", "psiblast",
        "-query", fasta_filename,
        "-db", args.psiblast_db_path,
        "-num_iterations", "3",
        "-out_ascii_pssm", pssm_filename,
        "-num_threads", "8"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument("--dataset", default='Testset')
    parser.add_argument("--folder", default='Testset')
    parser.add_argument("--hg", default='hg38')
    parser.add_argument("--prt_cut_len", default=100)
    parser.add_argument("--psiblast_db_path", default="./blast/db/swissprot/swissprot")
    parser.add_argument("--mkdssp_path", default="./bin/mkdssp")
    parser.add_argument("--step", default='obtain_mutilmodal_features',
                        choices=[
                                 'obtain_prt_cut',
                                 'obtain_mutilmodal_features',
                                 ])
    args = parser.parse_args()


    if args.step == 'obtain_prt_cut':
        asseq = "../out/prt_asseq/{}/{}_asseq".format(args.folder, args.dataset)
        variant = "../out/prt_asseq/{}/{}.exonic_variant_function".format(args.folder,args.dataset)
        obtain_prt_cut(asseq, variant, args)
    

    elif args.step == 'obtain_mutilmodal_features':
        obtain_mutilmodal_features(args)






  
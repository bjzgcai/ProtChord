# -*- coding: utf-8 -*-
import logging
import random
from tqdm import tqdm

import torch

logging.getLogger().setLevel(logging.ERROR)

import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import lmdb
import numpy as np
from rdkit import Chem
from torch.utils.data import Dataset
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

from protchord_core.data.prot_data.util import bstr2obj
from protchord_core.data.encoder_data.utils import VOCAB
from protchord_core.logging.loggers import logger


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
UNK_TOKEN_ID = 128257


def edge_index_to_adj(edge_index, num_atoms):
    adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    for i in range(edge_index.shape[1]):
        src, dest = edge_index[0, i], edge_index[1, i]
        adj_matrix[src, dest] = 1
        adj_matrix[dest, src] = 1
    return adj_matrix


def expand_adj_matrix(adj, a):
    b = adj.shape[0]
    ret_adj = np.ones((a, a), dtype=adj.dtype)
    ret_adj[a-b: , a-b:] = adj
    return ret_adj


def smiles2graph(smiles_string, removeHs=True, reorder_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
 
    mol = Chem.MolFromSmiles(smiles_string)
    mol = mol if removeHs else Chem.AddHs(mol)
    # if reorder_atoms:
    #     mol, _ = ReorderCanonicalRankAtoms(mol)
 
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)
 
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
 
            edge_feature = bond_to_feature_vector(bond)
 
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
 
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T
 
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)
 
    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
 
    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
 
    return graph



def psm_pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def psm_pad_adj_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(False)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = False
        x = new_x
    return x.unsqueeze(0)


def psm_pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(0.0)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def psm_pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def psm_pad_1d_chain_ids_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


class ComplexStageBCollator(object):
    def __init__(
        self,
        pad_token_id: int = 0,
        pp_mode: bool = True,
        protein_pad_id: int = 0, ##
        pos_pad_id: int = 0,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.pp_mode = pp_mode
        self.protein_pad_id = protein_pad_id
        self.pos_pad_id = pos_pad_id

    def pad_1d_unsqueeze(
        self, x: torch.Tensor, padlen: int, start: int, pad_num: Union[int, float]
    ):
        xlen = x.size(0)
        assert (
            start + xlen <= padlen
        ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
        new_x = x.new_full([padlen], pad_num, dtype=x.dtype)
        new_x[start : start + xlen] = x
        x = new_x

        return x.unsqueeze(0)

    def pad_pos_unsqueeze(self, x, padlen):
        xlen, xdim = x.size()
        if xlen < padlen:
            new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
            new_x[:xlen, :] = x
            x = new_x
        return x.unsqueeze(0)

    def __call__(self, samples_tuple):
        # ref_complex_samples, samples = samples_tuple
        sample_types = []
        input_ids = []
        labels = []
        psm_ids = []
        xyzs = []
        position_ids = []
        chain_ids = []
        adj_list = []
        node_attr_list = []
        in_degree = []
        num_atoms = []
        pocket_masks_list = []
        meta_info_list = []

        ref_sample_types = []
        ref_psm_ids = []
        ref_xyzs = []
        ref_position_ids = []
        ref_chain_ids = []
        ref_adj_list = []
        ref_node_attr_list = []
        ref_in_degree = []
        ref_num_atoms = []

        for (ref_sample, sample) in samples_tuple:
            sample_types.append(sample["sample_type"])
            input_ids.append(sample["input_ids"])
            labels.append(sample["labels"])
            psm_ids.append(sample["psm_ids"])
            xyzs.append(sample["xyzs"])
            position_ids.append(sample["position_ids"])
            chain_ids.append(sample["chain_ids"])
            adj_list.append(sample["adj"])
            node_attr_list.append(sample["node_attr"])
            in_degree.append(sample["adj"].sum(dim=0))
            num_atoms.append(len(sample["psm_ids"]))
            pocket_masks_list.append(sample["pocket_masks"])
            meta_info_list.append(sample["pl_tuple"])

            ref_sample_types.append(ref_sample["sample_type"])
            ref_psm_ids.append(ref_sample["psm_ids"])
            ref_xyzs.append(ref_sample["xyzs"])
            ref_position_ids.append(ref_sample["position_ids"])
            ref_chain_ids.append(ref_sample["chain_ids"])
            ref_adj_list.append(ref_sample["adj"])
            ref_node_attr_list.append(ref_sample["node_attr"])
            ref_in_degree.append(ref_sample["adj"].sum(dim=0))
            ref_num_atoms.append(len(ref_sample["psm_ids"]))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        if len(psm_ids) > 0:
            max_tokens = max(len(p) for p in psm_ids)
            # convert complex from list to torch tensor and pad it
            padded_complex = torch.cat([
                self.pad_1d_unsqueeze(torch.tensor(p), max_tokens, 0, self.protein_pad_id)
                for p in psm_ids
            ])
        else:
            padded_complex = torch.tensor([0], dtype=torch.int64).unsqueeze(0)
        
        if len(xyzs) > 0:
            max_tokens = max(len(p) for p in xyzs)  # add cls and eos token
            # convert proteins from list to torch tensor and pad it
            paded_pos = torch.cat([self.pad_pos_unsqueeze(torch.Tensor(pos), max_tokens) for pos in xyzs])
            if pocket_masks_list[0] is not None:
                paded_pocket_masks = torch.cat([self.pad_pos_unsqueeze(torch.Tensor(pocket_mask), max_tokens) for pocket_mask in pocket_masks_list])
            else:
                paded_pocket_masks = None
        else:
            # Need TO DO
            paded_pos = torch.tensor([0], dtype=torch.int64).unsqueeze(0)
            if pocket_masks_list[0] is not None:
                paded_pocket_masks = torch.tensor([0], dtype=torch.bool).unsqueeze(0)
            else:
                paded_pocket_masks = None
            max_tokens = 1

        batch_size = len(xyzs)

        if len(position_ids) > 0:
            position_ids = torch.cat([psm_pad_1d_unsqueeze(i, max_tokens) for i in position_ids])
        else:
            position_ids = torch.empty(0)

        if len(chain_ids) > 0:
            chain_ids = torch.cat([psm_pad_1d_chain_ids_unsqueeze(i, max_tokens) for i in chain_ids])
        else:
            chain_ids = torch.empty(0)

        if len(in_degree) > 0:
            in_degree = torch.cat([psm_pad_1d_unsqueeze(i, max_tokens) for i in in_degree])
        else:
            in_degree = torch.empty(0)
        
        ###### no use parameters
        attn_bias = torch.zeros((batch_size, max_tokens+1, max_tokens+1), dtype=torch.float)
        pbc = torch.zeros((batch_size, 4)).bool()
        cell = torch.zeros((batch_size, 3, 3), dtype=torch.float64)
        is_stable_periodic = torch.zeros((batch_size)).bool()
        node_type_edge = torch.zeros((batch_size, 3, 3)) # psm model will to process, but need to load
        # sample_type = [6] * batch_size # 2 for protein / 6 for complex
        energy = [[0.0]] * batch_size
        energy_per_atom = [[0.0]] * batch_size
        forces = torch.zeros((batch_size, max_tokens, 3), dtype=torch.float64)
        has_energy = torch.zeros((batch_size, 1), dtype=torch.float64)
        has_forces = torch.zeros((batch_size, 1), dtype=torch.float64)
        # if batch_size = 4
        # print(sample_type) # [6,6,6,6]
        # print(energy) # [[0.0], [0.0], [0.0], [0.0]]
        # print(energy_per_atom) # [[0.0], [0.0], [0.0], [0.0]]
        # print(forces) # shape = bsz * max_tokens * 3
        # print(has_energy) # tensor([ [0.], [0.], [0.], [0.] ], dtype=torch.float64)
        # print(has_forces) # tensor([ [0.], [0.], [0.], [0.] ], dtype=torch.float64)

        if len(adj_list) > 0:
            adj = torch.cat([psm_pad_attn_bias_unsqueeze(i, max_tokens) for i in adj_list])
        else:
            adj = torch.empty(0)
        if len(psm_ids) > 0:
            node_attr_list = [
                psm_pad_2d_unsqueeze(node_attr, max_tokens,)
                for node_attr in node_attr_list
            ]
            node_attr = torch.cat(node_attr_list)
        else:
            node_attr = torch.empty(0)

        # if len(meta_info_list) == 0:
        
        psm_batch_data = dict(
            sample_type=torch.Tensor(sample_types), # 1 for molecule / 2 for protein / 6 for complex
            attn_bias=attn_bias, # all 0.0 (max_tokens+1) TODO
            in_degree=in_degree, # from adj
            out_degree=in_degree,  # for undirected graph
            token_id=padded_complex, # protein_pad_id = 1
            node_attr=node_attr, #
            energy=torch.Tensor(energy), ###
            energy_per_atom=torch.Tensor(energy_per_atom), ###
            forces=forces, ###
            has_energy=has_energy, ###
            has_forces=has_forces, ###
            pos=paded_pos, # padded
            node_type_edge=node_type_edge, # TODO
            pbc=pbc, ### all false (for material)
            cell=cell, ### (for material)
            adj=adj, #
            num_atoms=torch.Tensor(num_atoms).long(), # num of item in complex (residue for protein and atom for ligand)
            is_stable_periodic=is_stable_periodic, ###
            position_ids=position_ids, # +1 because padding is 0
            chain_ids=chain_ids, # +1 because padding is 0
            pocket_masks=paded_pocket_masks,
            # meta_info=torch.empty(0),
            meta_info=meta_info_list,
        )
        # print(psm_batch_data)
        
        # ref_sample_types = [6] * batch_size

        if len(ref_psm_ids) > 0:
            ref_max_tokens = max(len(p) for p in ref_psm_ids)
            # convert complex from list to torch tensor and pad it
            ref_padded_complex = torch.cat([
                self.pad_1d_unsqueeze(torch.tensor(p), ref_max_tokens, 0, self.protein_pad_id)
                for p in ref_psm_ids
            ])
        else:
            ref_padded_complex = torch.tensor([0], dtype=torch.int64).unsqueeze(0)
        
        if len(ref_xyzs) > 0:
            ref_max_tokens = max(len(p) for p in ref_xyzs)  # add cls and eos token
            # convert proteins from list to torch tensor and pad it
            ref_paded_pos = torch.cat([self.pad_pos_unsqueeze(torch.Tensor(pos), ref_max_tokens) for pos in ref_xyzs])
        else:
            # Need TO DO
            ref_paded_pos = torch.tensor([0], dtype=torch.int64).unsqueeze(0)
            ref_max_tokens = 1
        
        ref_paded_pocket_masks = None

        if len(ref_position_ids) > 0:
            ref_position_ids = torch.cat([psm_pad_1d_unsqueeze(i, ref_max_tokens) for i in ref_position_ids])
        else:
            ref_position_ids = torch.empty(0)

        if len(ref_chain_ids) > 0:
            ref_chain_ids = torch.cat([psm_pad_1d_chain_ids_unsqueeze(i, ref_max_tokens) for i in ref_chain_ids])
        else:
            ref_chain_ids = torch.empty(0)

        if len(ref_in_degree) > 0:
            ref_in_degree = torch.cat([psm_pad_1d_unsqueeze(i, ref_max_tokens) for i in ref_in_degree])
        else:
            ref_in_degree = torch.empty(0)
        
        if len(ref_adj_list) > 0:
            ref_adj = torch.cat([psm_pad_attn_bias_unsqueeze(i, ref_max_tokens) for i in ref_adj_list])
        else:
            ref_adj = torch.empty(0)
        
        if len(ref_psm_ids) > 0:
            ref_node_attr_list = [
                psm_pad_2d_unsqueeze(node_attr, ref_max_tokens,)
                for node_attr in ref_node_attr_list
            ]
            ref_node_attr = torch.cat(ref_node_attr_list)
        else:
            ref_node_attr = torch.empty(0)
        
        ref_attn_bias = torch.zeros((batch_size, ref_max_tokens+1, ref_max_tokens+1), dtype=torch.float)
        ref_pbc = torch.zeros((batch_size, 4)).bool()
        ref_cell = torch.zeros((batch_size, 3, 3), dtype=torch.float64)
        ref_is_stable_periodic = torch.zeros((batch_size)).bool()
        ref_node_type_edge = torch.zeros((batch_size, 3, 3)) # psm model will to process, but need to load
        # sample_type = [6] * batch_size # 2 for protein / 6 for complex
        ref_energy = [[0.0]] * batch_size
        ref_energy_per_atom = [[0.0]] * batch_size
        ref_forces = torch.zeros((batch_size, ref_max_tokens, 3), dtype=torch.float64)
        ref_has_energy = torch.zeros((batch_size, 1), dtype=torch.float64)
        ref_has_forces = torch.zeros((batch_size, 1), dtype=torch.float64)
        
        ref_psm_batch_data = dict(
            sample_type=torch.Tensor(ref_sample_types), # 1 for molecule / 2 for protein / 6 for complex
            attn_bias=ref_attn_bias, # all 0.0 (ref_max_tokens+1)
            in_degree=ref_in_degree, # from adj
            out_degree=ref_in_degree,  # for undirected graph
            token_id=ref_padded_complex, # protein_pad_id = 1
            node_attr=ref_node_attr, #
            energy=torch.Tensor(ref_energy), ###
            energy_per_atom=torch.Tensor(ref_energy_per_atom), ###
            forces=ref_forces, ###
            has_energy=ref_has_energy, ###
            has_forces=ref_has_forces, ###
            pos=ref_paded_pos, # padded
            node_type_edge=ref_node_type_edge,
            pbc=ref_pbc, ### all false (for material)
            cell=ref_cell, ### (for material)
            adj=ref_adj, #
            num_atoms=torch.Tensor(ref_num_atoms).long(), # num of item in complex (residue for protein and atom for ligand)
            is_stable_periodic=ref_is_stable_periodic, ###
            position_ids=ref_position_ids, # +1 because padding is 0
            chain_ids=ref_chain_ids, # +1 because padding is 0
            pocket_masks=ref_paded_pocket_masks,
        )


        if self.pp_mode or True:
            input_tuple = (
                input_ids,
                labels,
                input_ids.ne(self.pad_token_id),
                psm_batch_data,
                ref_psm_batch_data,
            )
            label_tuple = (labels, input_ids.ne(self.pad_token_id))
            return (input_tuple, label_tuple)
        else:
            pass


class ComplexStageBDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        model_max_length: int,
        protein_max_size: int,
        pad_token_id: int,
        max_pro_per_sample: int = 2,
        pool_mode: Optional[str] = "full",
        embedding_length: int = 20,
        num_token_id: int = 32003,
        protein_pad_id: int = 0,
        pos_pad_id: int = 0,
        pp_mode: bool = True,
        local_rank: int = 0,
        use_llama_tokenizer: bool = False,
        instruction_mode: bool = False,
        psm_max_length: int = 2048,
        use_vae: bool = False,
        gearnet_graph_lmdb_path: str = "",
        use_dpo: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.psm_max_length = psm_max_length
        self.model_max_length = model_max_length
        self.protein_max_size = protein_max_size
        self.pool_mode = pool_mode
        self.embedding_length = embedding_length
        self.num_token_id = num_token_id
        self.local_rank = local_rank
        self.use_vae = use_vae
        self.use_dpo = use_dpo

        self.max_pro_per_sample = max_pro_per_sample

        self.len = 0
        # self.index_to_key_map = []
        # self.read_txns = {}
        # self.read_envs = {}
        # self.dataset_count = {}
        # self.dataset_filtered = {}

        self.env = lmdb.open(
            str(self.data_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self.len, self.keys = metadata["size"], metadata["keys"]

        self.gearnet_graph_txn = None
        if gearnet_graph_lmdb_path and gearnet_graph_lmdb_path.lower() != "no":
            self.gearnet_graph_env = lmdb.open(
                str(gearnet_graph_lmdb_path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
            )
            self.gearnet_graph_txn = self.gearnet_graph_env.begin(write=False)
            logger.info(f"Use gearnet precompute lmdb: {gearnet_graph_lmdb_path}")

        self.collate_fn = ComplexStageBCollator(
            pad_token_id=pad_token_id,
            pp_mode=pp_mode,
            protein_pad_id=protein_pad_id,
            pos_pad_id=pos_pad_id,
        )
        self.instruction_mode = instruction_mode
        self.keys, self.len_filter = self.filter_dataset()

        logger.info(f"Dataset size: {self.len}, Filtered size: {self.len_filter}")
        self.len = self.len_filter

        self.use_llama_tokenizer = use_llama_tokenizer

    def filter_dataset(self):
        filter_keys = []
        for key in tqdm(self.keys):
            value = self.txn.get(str(key).encode())
            if value is None:
                raise IndexError(f"Name {key} has no data in the dataset")
            data_tuple = pkl.loads(value)
            if len(data_tuple) == 8:
                sample_type, input_ids, original_labels, protein_nlm_ids, \
                psm_token_num, protein_tokens, protein_xyzs, extra_info = data_tuple
                ref_ligand_atom_num = 0
            elif len(data_tuple) == 9:
                sample_type, input_ids, original_labels, protein_nlm_ids, \
                psm_token_num, protein_tokens, protein_xyzs, extra_info, pocket_masks = data_tuple
                ref_ligand_atom_num = 0
            elif len(data_tuple) == 10:
                sample_type, input_ids, original_labels, protein_nlm_ids, \
                psm_token_num, protein_tokens, protein_xyzs, extra_info, ref_ligand, pocket_masks = data_tuple
                ref_ligand_atom_num = 0
                if len(ref_ligand) == 3:
                    ref_ligand_atom_num = ref_ligand[2]
            
            len_protein = sum([len(p) for p in protein_tokens])

            skip = 0
            if sample_type==2 or sample_type==6:
                ## for target to drug
                # if len(protein_nlm_ids) > self.protein_max_size or len(protein_nlm_ids) < 4: # pocket
                #     print(f"Reference pocket length not in [4, {self.protein_max_size}]! pocket_nlm_ids: {protein_nlm_ids}")
                #     skip = 1
                #     continue

                if len_protein > self.protein_max_size or len_protein < 4:
                    print(f"Reference protein length not in [4, {self.protein_max_size}]! protein_tokens: {protein_tokens}")
                    skip = 1
                    continue
            
                if self.use_vae:
                    if len_protein + ref_ligand_atom_num > self.protein_max_size:
                        print(f"Reference complex exceed max length! protein_nlm_ids: {protein_nlm_ids}, ref_ligand_atom_num: {ref_ligand_atom_num}")
                        skip = 1
                        continue
            
            sample_len = len(input_ids)
            if sample_type == 1:
                sample_len += psm_token_num - 1
            elif sample_type == 2:
                sample_len += psm_token_num - 1
            elif sample_type == 6:
                sample_len += psm_token_num - 2
            
            if sample_len > self.model_max_length:
                print(f"sample_len: {sample_len}, input_ids: {input_ids}, protein_nlm_ids: {protein_nlm_ids}")
                skip = 1
                continue

            if not skip:
                filter_keys.append(key)

        return filter_keys, len(filter_keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        key = self.keys[index]
        value = self.txn.get(str(key).encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data_tuple = pkl.loads(value)
        sample_type = data_tuple[0]
        # if sample_type == 1:
        #     ret = self._getitem_molecule(data_tuple)
        if sample_type == 2:
            ret = self._getitem_protein(data_tuple)
        if sample_type == 6:
            ret = self._getitem_complex(data_tuple)
        return ret

    # def _getitem_molecule(self, data_tuple):
    #     if len(data_tuple) == 8:
    #         sample_type, input_ids, original_labels, _, \
    #         psm_token_num, atoms, xyzs, extra_info = data_tuple
    #         pocket_masks = None
    #     else:
    #         logger.info("Data error!!!")
        
    #     if isinstance(original_labels, list):
    #         original_labels = torch.tensor(original_labels, dtype=torch.int64)
    #     if isinstance(input_ids, list):
    #         input_ids = torch.tensor(input_ids, dtype=torch.int64)

    #     smiles = extra_info[1]
        
    #     mol_graph = smiles2graph(smiles)
    #     mol_num_atoms = mol_graph['num_nodes']
    #     mol_adj = edge_index_to_adj(mol_graph['edge_index'], mol_num_atoms)
    #     mol_node_attr = mol_graph['node_feat']
    #     mol_psm_ids = (mol_node_attr[:, 0] + 2).tolist()
    #     mol_psm_ids = [int(i) for i in mol_psm_ids]
    #     assert len(mol_psm_ids) == mol_num_atoms, f"mol psm ids length {len(mol_psm_ids)} != num atoms {mol_num_atoms}"

    #     start_position_id = 0
    #     position_ids = list(range(start_position_id, start_position_id + mol_num_atoms))
    #     start_position_id += mol_num_atoms + 1000
        
    #     cur_chain_id = 1
    #     chain_ids = [cur_chain_id] * mol_num_atoms
    #     cur_chain_id += 1
        
    #     new_xyzs = xyzs

    #     mol_psm_ids = torch.tensor(mol_psm_ids)
    #     position_ids = torch.tensor(position_ids)
    #     chain_ids = torch.tensor(chain_ids)

    #     new_input_ids = []
    #     labels = []
    #     # print(input_ids)
    #     if self.pool_mode == "full":
    #         new_input_ids = []
    #         for input_id in input_ids:
    #             if input_id == -2:
    #                 new_input_ids.extend([-1]*len(mol_psm_ids))
    #             else:
    #                 new_input_ids.append(input_id)
            
    #         input_ids = new_input_ids
    #         labels = original_labels
    #         if len(input_ids) != len(labels):
    #             print(f"input_ids: {input_ids}")
    #             print(f"labels: {labels}")
    #         assert len(input_ids) == len(labels), f"Length mismatch: extra_info {extra_info}."

    #     elif self.pool_mode == "qformer":
    #         raise NotImplementedError
    #     else:
    #         raise ValueError(f"Invalid pool_mode: {self.pool_mode}")

    #     if original_labels is not None and not isinstance(original_labels, tuple):
    #         label = original_labels[-len(input_ids) :]
    #     input_ids = torch.tensor(new_input_ids).to(dtype=torch.int64)
    #     labels = torch.tensor(label).to(dtype=torch.int64)

    #     mol_adj = torch.tensor(mol_adj, dtype=torch.bool)
    #     mol_node_attr = torch.tensor(mol_node_attr, dtype=torch.long)

    #     # print(f"len(input_ids): {len(input_ids)}")
    #     # print(f"len(labels): {len(labels)}")
    #     # print(f"len(psm_ids): {len(mol_psm_ids)}")
    #     # print(f"len(xyzs): {len(new_xyzs)}")
    #     # print(f"len(position_ids): {len(position_ids)}")
    #     # print(f"len(chain_ids): {len(chain_ids)}")
    #     # print(f"size(adj): {mol_adj.size()}")
    #     # print(f"size(node_attr): {mol_node_attr.size()}")

    #     return dict(
    #         sample_type=sample_type,
    #         input_ids=input_ids,
    #         labels=labels,
    #         psm_ids=mol_psm_ids,
    #         xyzs=new_xyzs,
    #         position_ids=position_ids,
    #         chain_ids=chain_ids,
    #         adj=mol_adj,
    #         node_attr=mol_node_attr,
    #         pocket_masks=None,
    #     )

    def _getitem_protein(self, data_tuple):
        if len(data_tuple) == 8:
            sample_type, input_ids, original_labels, protein_nlm_ids, \
            psm_token_num, protein_tokens, protein_xyzs, extra_info = data_tuple
            pocket_masks = None
        elif len(data_tuple) == 9:
            sample_type, input_ids, original_labels, protein_nlm_ids, \
            psm_token_num, protein_tokens, protein_xyzs, extra_info, pocket_masks = data_tuple # if stageB, protein_nlm_ids is pocket_nlm_ids
        elif len(data_tuple) == 10:
            sample_type, input_ids, original_labels, protein_nlm_ids, \
            psm_token_num, protein_tokens, protein_xyzs, extra_info, ref_ligand_info, pocket_masks = data_tuple
        else:
            logger.info("Data error!!!")
        
        if len(extra_info) == 3:
            pl_tuple = extra_info[-1]
        elif len(extra_info) == 4:
            pl_tuple = extra_info[-1] # desc
        else:
            pl_tuple = None
        
        if isinstance(original_labels, list):
            original_labels = torch.tensor(original_labels, dtype=torch.int64)

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
        cur_proteins = []
        position_ids = []
        chain_ids = []
        pocket_mask_list = []
        start_position_id = 0
        cur_chain_id = 1
        new_xyzs = np.zeros((0, 3))
 
        if isinstance(protein_xyzs, list):
            for i in range(len(protein_xyzs)):
                new_xyzs = np.concatenate((new_xyzs, protein_xyzs[i]), axis=0)
        else:
            new_xyzs = np.concatenate((new_xyzs, protein_xyzs), axis=0)

        for i, pro in enumerate(protein_tokens):
            cur_proteins.extend(pro)
            if pocket_masks is not None:
                pocket_mask_list.extend(pocket_masks[i])
            position_ids.extend(range(start_position_id, start_position_id + len(pro)))
            chain_ids.extend([cur_chain_id] * len(pro))
            cur_chain_id += 1
            start_position_id += len(pro)+1000
        
        protein_psm_ids = [VOCAB[tok] for tok in cur_proteins]
        protein_psm_ids_tensor = torch.tensor(protein_psm_ids, dtype=torch.long)
        protein_node_attr = torch.cat([protein_psm_ids_tensor.unsqueeze(-1), torch.ones((len(protein_psm_ids), 8), dtype=torch.long)], dim=1)

        new_input_ids = []
        labels = []
        if self.pool_mode == "full":
            new_input_ids = []
            for input_id in input_ids:
                if input_id == -1:
                    if pocket_masks is None: # for stageA
                        new_input_ids.extend([-1]*len(protein_psm_ids))
                    else: # for stageB
                        new_input_ids.extend([-1]*psm_token_num)
                else:
                    new_input_ids.append(input_id)
            
            input_ids = new_input_ids
            labels = original_labels
            assert len(input_ids) == len(labels), f"Length mismatch: extra_info {extra_info}."

        elif self.pool_mode == "qformer":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid pool_mode: {self.pool_mode}")

        # if original_labels is not None and not isinstance(original_labels, tuple):
        #     label = original_labels[-len(input_ids) :]
        input_ids = torch.tensor(new_input_ids).to(dtype=torch.int64)
        labels = torch.tensor(labels).to(dtype=torch.int64)
        protein_adj = np.ones((len(protein_psm_ids), len(protein_psm_ids)), dtype=int)
        protein_adj = torch.tensor(protein_adj, dtype=torch.bool)
        
        if pocket_masks is not None:
            pocket_masks = torch.tensor(pocket_mask_list, dtype=torch.bool)
            true_count = torch.sum(pocket_masks).item()
            assert true_count == psm_token_num, f"pocket mask true count {true_count} != psm token num {psm_token_num}"
        else:
            pocket_masks = None
        
        protein_ref_dict = dict(
            sample_type=2,
            psm_ids=torch.tensor(protein_psm_ids, dtype=torch.long),
            xyzs=new_xyzs,
            position_ids=torch.tensor(position_ids, dtype=torch.long),
            chain_ids=torch.tensor(chain_ids, dtype=torch.long),
            adj=protein_adj,
            node_attr=protein_node_attr,
        )

        protein_psm_ids = torch.tensor(protein_psm_ids)
        position_ids = torch.tensor(position_ids)
        chain_ids = torch.tensor(chain_ids)
        
        return (protein_ref_dict, dict(
            sample_type=sample_type,
            input_ids=input_ids,
            labels=labels,
            psm_ids=protein_psm_ids,
            xyzs=new_xyzs,
            position_ids=position_ids,
            chain_ids=chain_ids,
            adj=protein_adj,
            node_attr=protein_node_attr,
            pocket_masks=pocket_masks,
            pl_tuple=pl_tuple,
        ))

    def _getitem_complex(self, data_tuple):
        if len(data_tuple) == 8:
            sample_type, input_ids, original_labels, protein_nlm_ids, \
            psm_token_num, protein_tokens, protein_xyzs, extra_info = data_tuple
            pocket_masks = None
        elif len(data_tuple) == 9:
            sample_type, input_ids, original_labels, protein_nlm_ids, \
            psm_token_num, protein_tokens, protein_xyzs, extra_info, pocket_masks = data_tuple # if stageB, protein_nlm_ids is pocket_nlm_ids
        elif len(data_tuple) == 10:
            sample_type, input_ids, original_labels, protein_nlm_ids, \
            psm_token_num, protein_tokens, protein_xyzs, extra_info, ref_ligand_info, pocket_masks = data_tuple
        else:
            logger.info("Data error!!!")
        
        if isinstance(original_labels, list):
            original_labels = torch.tensor(original_labels, dtype=torch.int64)

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
        
        cur_proteins = []
        position_ids = []
        chain_ids = []
        start_position_id = 0
        cur_chain_id = 1
        new_xyzs = np.zeros((0, 3))
        new_xyzs = np.concatenate((new_xyzs, protein_xyzs), axis=0)

        for i, pro in enumerate(protein_tokens):
            cur_proteins.extend(pro)
            position_ids.extend(range(start_position_id, start_position_id + len(pro)))
            chain_ids.extend([cur_chain_id] * len(pro))
            cur_chain_id += 1
            start_position_id += len(pro)+1000
        
        protein_psm_ids = [VOCAB[tok] for tok in cur_proteins]
        protein_psm_ids_tensor = torch.tensor(protein_psm_ids, dtype=torch.long)
        protein_node_attr = torch.cat([protein_psm_ids_tensor.unsqueeze(-1), torch.ones((len(protein_psm_ids), 8), dtype=torch.long)], dim=1)

        smiles = extra_info[1]
        mol_graph = smiles2graph(smiles)
        mol_num_atoms = mol_graph['num_nodes']
        mol_adj = edge_index_to_adj(mol_graph['edge_index'], mol_num_atoms)
        mol_node_attr = mol_graph['node_feat']
        mol_psm_ids = (mol_node_attr[:, 0] + 2).tolist()
        mol_psm_ids = [int(i) for i in mol_psm_ids]

        assert len(mol_psm_ids) == mol_num_atoms, f"mol psm ids length {len(mol_psm_ids)} != num atoms {mol_num_atoms}"

        complex_psm_ids = protein_psm_ids + mol_psm_ids
        complex_position_ids = position_ids + list(range(start_position_id, start_position_id + mol_num_atoms))
        complex_chain_ids = chain_ids + [cur_chain_id] * mol_num_atoms
        complex_xyzs = new_xyzs

        complex_psm_ids = torch.tensor(complex_psm_ids)
        complex_position_ids = torch.tensor(complex_position_ids)
        complex_chain_ids = torch.tensor(complex_chain_ids)

        new_input_ids = []
        labels = []
        # print(input_ids)
        if self.pool_mode == "full":
            new_input_ids = []
            for input_id in input_ids:
                if input_id == -1:
                    new_input_ids.extend([-1]*len(protein_psm_ids))
                elif input_id == -2:
                    new_input_ids.extend([-1]*mol_num_atoms)
                else:
                    new_input_ids.append(input_id)
            
            input_ids = new_input_ids
            labels = original_labels
            assert len(input_ids) == len(labels), f"Length mismatch: extra info {extra_info}."

        elif self.pool_mode == "qformer":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid pool_mode: {self.pool_mode}")

        if original_labels is not None and not isinstance(original_labels, tuple):
            label = original_labels[-len(input_ids) :]
        input_ids = torch.tensor(new_input_ids).to(dtype=torch.int64)
        labels = torch.tensor(label).to(dtype=torch.int64)

        complex_num = len(protein_psm_ids) + mol_num_atoms
        complex_num = min(complex_num, self.psm_max_length)
        complex_adj = expand_adj_matrix(mol_adj, complex_num)
        complex_adj = torch.tensor(complex_adj, dtype=torch.bool)
        mol_node_attr = torch.tensor(mol_node_attr, dtype=torch.long)
        complex_node_attr = torch.cat([protein_node_attr, mol_node_attr], dim=0)

        return dict(
            sample_type=sample_type,
            input_ids=input_ids,
            labels=labels,
            psm_ids=complex_psm_ids,
            xyzs=complex_xyzs,
            position_ids=complex_position_ids,
            chain_ids=complex_chain_ids,
            adj=complex_adj,
            node_attr=complex_node_attr,
            pocket_masks=pocket_masks,
        )

    def collate(self, samples):
        return self.collate_fn(samples)

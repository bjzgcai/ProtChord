# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional



@dataclass
class GraphormerConfig:
    model_type: str = "graphormer"
    num_classes: int = 1
    encoder_attention_heads: int = 32
    encoder_ffn_embed_dim: int = 768
    encoder_embed_dim: int = 768
    encoder_layers: int = 24
    num_pred_attn_layer: int = 4
    num_3d_bias_kernel: int = 128
    max_length: int = 1024
    pbc_expanded_token_cutoff: int = 512
    pbc_expanded_num_cell_per_direction: int = 10
    multi_hop_max_dist: int = 20

    droppath_prob: float = 0.0
    act_dropout: float = 0.0
    attn_dropout: float = 0.0
    dropout: float = 0.0
    sandwich_ln: bool = True
    noise_scale: float = 0.2
    mask_ratio: float = 0.5
    d_tilde: float = 1.0
    pbc_cutoff: float = 40.0

    data_path: str = ""
    dataset_names: str = ""
    loadcheck_path: str = ""

    add_3d: bool = False
    no_2d: bool = False
    ft: bool = False
    infer: bool = False
    use_pbc: bool = False

    # TM and ddpm params
    transformer_m_pretrain: bool = True
    mode_prob: str = "0.6,0.2,0.2"
    num_timesteps: int = 5000
    num_timesteps_stepsize: int = -1
    ddpm_beta_start: float = 1e-7
    ddpm_beta_end: float = 0.002
    ddpm_schedule: str = "sigmoid"
    noise_mode: str = "const"
    num_edges: int = 25600
    num_atom_features: int = 5120
    ###########################################################################################
    ####### THESE are from graphormer_base_architecture, is confict, follow upper default vaule.
    #
    #
    #     dropout: 0.0
    attention_dropout: float = 0.0
    #     act_dropout = 0.0
    #
    #     encoder_ffn_embed_dim = 4096
    #     encoder_layers = 6
    #     encoder_attention_heads = 8
    #
    #     encoder_embed_dim = 1024
    share_encoder_input_output_embed: bool = False
    encoder_learned_pos: bool = False
    no_token_positional_embeddings: bool = False
    num_segments: int = 2
    #
    sentence_class_num: int = 2
    sent_loss: bool = False
    #
    apply_bert_init: bool = False
    #
    activation_fn: str = "relu"
    pooler_activation_fn: str = "tanh"
    encoder_normalize_before: bool = False
    #
    #     sandwich_ln = False
    #     droppath_prob = 0.0
    #
    # add
    atom_loss_coeff: float = 1.0
    pos_loss_coeff: float = 1.0
    y_2d_loss_coeff: float = 1.0
    #
    max_positions: int = 512
    num_atoms: int = 512 * 9
    #     num_edges = 512 * 3
    num_in_degree: int = 512
    num_out_degree: int = 512
    num_spatial: int = 512
    num_edge_dis: int = 128
    #     multi_hop_max_dist:int = 5
    edge_type: str = "multi_hop"
    #
    layerdrop: float = 0.0
    apply_graphormer_init: bool = True
    pre_layernorm: bool = False
    #
    #
    #
    #     encoder_layers = 12
    #     encoder_attention_heads = 12
    #     encoder_ffn_embed_dim = 768
    #     encoder_embed_dim = 768
    #     dropout = 0.0
    #     attention_dropout = 0.0
    #     act_dropout = 0.0
    #     activation_fn ="gelu"
    #     encoder_normalize_before = True
    #     share_encoder_input_output_embed = False
    #     no_token_positional_embeddings = False
    #
    #
    #

    def __init__(
        self,
        args,
        **kwargs,
    ):
        graphormer_base_architecture(args)

        if not hasattr(args, "max_positions"):
            try:
                args.max_positions = args.tokens_per_sample
            except:
                args.max_positions = args.max_nodes

        # set attributes of args to self
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))

        self.num_atoms = args.num_atoms
        self.num_in_degree = args.num_in_degree
        self.num_out_degree = args.num_out_degree
        self.num_edges = args.num_edges
        self.num_spatial = args.num_spatial
        self.num_edge_dis = args.num_edge_dis
        self.edge_type = args.edge_type
        self.multi_hop_max_dist = args.multi_hop_max_dist
        self.num_encoder_layers = args.encoder_layers
        self.embedding_dim = args.encoder_embed_dim
        self.ffn_embedding_dim = args.encoder_ffn_embed_dim
        self.num_attention_heads = args.encoder_attention_heads
        self.dropout = args.dropout
        self.attention_dropout = args.attention_dropout
        self.activation_dropout = args.act_dropout
        self.layerdrop = args.layerdrop
        self.max_seq_len = args.max_positions
        self.num_segments = args.num_segments
        self.use_position_embeddings = not args.no_token_positional_embeddings
        self.encoder_normalize_before = args.encoder_normalize_before
        self.apply_bert_init = args.apply_bert_init
        self.activation_fn = args.activation_fn
        self.learned_pos_embedding = args.encoder_learned_pos
        self.sandwich_ln = args.sandwich_ln
        self.droppath_prob = args.droppath_prob
        self.add_3d = args.add_3d
        self.num_3d_bias_kernel = args.num_3d_bias_kernel
        self.no_2d = args.no_2d
        self.args = args


def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.num_segments = getattr(args, "num_segments", 2)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)

    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)

    # add
    args.atom_loss_coeff = getattr(args, "atom_loss_coeff", 1.0)
    args.pos_loss_coeff = getattr(args, "pos_loss_coeff", 1.0)
    args.y_2d_loss_coeff = getattr(args, "y_2d_loss_coeff", 1.0)

    args.max_positions = getattr(args, "max_positions", 512)
    args.num_atoms = getattr(args, "num_atoms", 512 * 9)
    args.num_edges = getattr(args, "num_edges", 512 * 3)
    args.num_in_degree = getattr(args, "num_in_degree", 512)
    args.num_out_degree = getattr(args, "num_out_degree", 512)
    args.num_spatial = getattr(args, "num_spatial", 512)
    args.num_edge_dis = getattr(args, "num_edge_dis", 128)
    args.multi_hop_max_dist = getattr(args, "multi_hop_max_dist", 5)
    args.edge_type = getattr(args, "edge_type", "multi_hop")

    args.layerdrop = getattr(args, "layerdrop", 0.0)

    # args.q_noise = getattr(args, "q_noise", 0.0)
    # args.qn_block_size = getattr(args, "qn_block_size", 8)
    # args.freeze_embeddings = getattr(args, "freeze_embeddings", False)
    # args.n_trans_layers_to_freeze = getattr(args, "n_trans_layers_to_freeze", 0)
    # args.export = getattr(args, "export", False)
    # args.export = getattr(args, "export", False)
    # args.traceable = getattr(args, "traceable", False)
    # args.embed_scale = getattr(args, "embed_scale", None)
    # args.apply_bert_init = getattr(args, "apply_bert_init", False)


def graphormer_base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.act_dropout = getattr(args, "act_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)


class VecInitApproach(Enum):
    ZERO_CENTERED_POS: str = "ZERO_CENTERED_POS"
    RELATIVE_POS: str = "RELATIVE_POS"
    AUGMENTED_RELATIVE_POS: str = "AUGMENTED_RELATIVE_POS"
    RELATIVE_POS_VEC_BIAS: str = "RELATIVE_POS_VEC_BIAS"

    def __str__(self):
        return self.value


class DiffusionTrainingLoss(Enum):
    L1: str = "L1"
    L2: str = "L2"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"

    def __str__(self):
        return self.value


class ForceLoss(Enum):
    L1: str = "L1"
    L2: str = "L2"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"
    NoiseTolerentL1: str = "NoiseTolerentL1"

    def __str__(self):
        return self.value


class DiffusionTimeStepEncoderType(Enum):
    DISCRETE_LEARNABLE: str = "DISCRETE_LEARNABLE"
    POSITIONAL: str = "POSITIONAL"

    def __str__(self):
        return self.value


class ForceHeadType(Enum):
    LINEAR: str = "LINEAR"
    GATED_EQUIVARIANT: str = "GATED_EQUIVARIANT"
    MLP: str = "MLP"

    def __str__(self) -> str:
        return self.value


class GaussianFeatureNodeType(Enum):
    EXCHANGABLE: str = "EXCHANGABLE"
    NON_EXCHANGABLE: str = "NON_EXCHANGABLE"

    def __str__(self) -> str:
        return self.value


@dataclass
class EncoderConfig(GraphormerConfig):
    model_type: str = "psm"
    seq_masking_method: str = "transformerM"

    add_rope: bool = True
    rope_theta: int = 10000
    num_residues: int = 32
    max_num_aa: int = 1024

    encoder_pair_embed_dim: int = 32
    decoder_ffn_dim: int = 2048
    decoder_hidden_dim: int = 512

    task: str = "mae"
    sample_mode: bool = False

    train_data_path: str = ""
    valid_data_path: str = ""

    data_path_list: str = ""
    dataset_name_list: str = ""
    dataset_split_raito: str = ""
    dataset_micro_batch_size: str = ""

    lamb_pde: float = 0.01

    # for PBC
    pbc_expanded_token_cutoff: int = 512
    pbc_expanded_num_cell_per_direction: int = 5
    pbc_expanded_distance_cutoff: float = 20.0
    pbc_use_local_attention: bool = False
    pbc_multigraph_cutoff: float = 5.0
    diff_init_lattice_size_factor: float = 2.859496852322873
    diff_init_lattice_size: float = 10.0
    use_fixed_init_lattice_size: bool = False
    add_unit_cell_virtual_node: bool = False

    # for protein
    crop_radius: float = 25.0
    mode_prob: str = "0.1,0.4,0.5"
    complex_mode_prob: str = "0.1,0.4,0.5"
    sample_ligand_only: bool = False
    plddt_threshold: float = 70.0

    # for molecule
    molecule_ref_energy_source: Optional[str] = None
    molecule_outlier_energy_atoms: str = "DEPRECATED_PM6_ATOM_ENERGY_OUTLIER_LIST"

    # for diffusion
    diffusion_sampling: str = "ddpm"
    num_timesteps_stepsize: int = -1
    diffusion_mode: str = "epsilon"
    diffusion_noise_std: float = 1.0
    use_adaptive_noise_std_for_periodic: bool = False
    periodic_diffusion_noise_std_factor: float = 1.0531306506190654
    periodic_lattice_diffusion_noise_std: float = 0.5
    diffusion_rescale_coeff: float = 1.0
    ddim_eta: float = 0.0
    ddim_steps: int = 50
    clean_sample_ratio: float = 0.5
    diffusion_training_loss: DiffusionTrainingLoss = DiffusionTrainingLoss.MSE
    diffusion_time_step_encoder_type: DiffusionTimeStepEncoderType = (
        DiffusionTimeStepEncoderType.POSITIONAL
    )
    align_x0_in_diffusion_loss: bool = True
    separate_noise_head: bool = False

    # EDM
    edm_P_mean: float = -1.2
    edm_P_std: float = 1.5
    edm_sigma_data: float = 16.0
    edm_sample_num_steps: int = 200
    edm_sample_sigma_min: float = 0.004
    edm_sample_sigma_max: float = 160.0
    edm_sample_rho: float = 7.0
    edm_sample_S_churn: float = 0.0
    edm_sample_S_min: float = 0.0
    edm_sample_S_max: float = 3.0e30
    edm_sample_S_noise: float = 1.0
    # for AF3
    af3_sample_gamma_0: float = 0.8
    af3_sample_gamma_min: float = 1.0
    af3_sample_step_scale: float = 1.5
    noise_embedding: str = "fourier"

    # for RL
    psm_finetuneRL_mode: bool = True
    diffusion_sampling_rl: str = "ddpm"
    num_timesteps_stepsize_rl: int = -1
    reward_model: str = "lddt"
    psm_value_step: int = 1
    perturbation_each_traj: int = 2
    reward_weight: float = 10.0
    kl_weight: float = 0.1
    ratio_clip: float = 1e-4

    # EDM
    edm_P_mean: float = -1.2
    edm_P_std: float = 1.5
    edm_sigma_data: float = 16.0
    edm_sample_num_steps: int = 200
    edm_sample_sigma_min: float = 4e-4  # 0.004
    edm_sample_sigma_max: float = 160.0
    edm_sample_rho: float = 7.0
    edm_sample_S_churn: float = 0.0
    edm_sample_S_min: float = 0.0
    edm_sample_S_max: float = 100  # 3.0e30
    edm_sample_S_noise: float = 1.0
    # for AF3
    af3_sample_gamma_0: float = 0.8
    af3_sample_gamma_min: float = 1.0
    af3_sample_step_scale: float = 1.5
    noise_embedding: str = "fourier"
    # for force
    force_loss_type: ForceLoss = ForceLoss.L1
    force_head_type: ForceHeadType = ForceHeadType.GATED_EQUIVARIANT
    node_type_edge_method: GaussianFeatureNodeType = (
        GaussianFeatureNodeType.NON_EXCHANGABLE
    )

    # for equivariant part
    equivar_vec_init: VecInitApproach = VecInitApproach.ZERO_CENTERED_POS
    equivar_use_linear_bias: bool = False
    equivar_use_attention_bias: bool = False
    use_smooth_softmax: bool = False
    smooth_factor: float = 20.0
    use_smooth_equviariant_norm: bool = True
    no_rotary_embedding_for_vector: bool = False
    mlm_from_decoder_feature: bool = False
    disable_data_aug: bool = False
    use_fp32_in_decoder: bool = False

    # for 2D information
    use_2d_atom_features: bool = True
    use_2d_bond_features: bool = False
    preprocess_2d_bond_features_with_cuda: bool = True
    share_attention_bias: bool = False

    # memory efficient attention
    use_memory_efficient_attention: bool = False

    # loss computation options
    rescale_loss_with_std: bool = True
    material_force_loss_ratio: float = 1.0
    material_energy_loss_ratio: float = 1.0
    molecule_force_loss_ratio: float = 1.0
    molecule_energy_loss_ratio: float = 1.0
    energy_per_atom_label_scale: float = 1.0
    molecule_energy_per_atom_std_override: float = 1.0
    decoder_feat4energy: bool = True
    encoderfeat4noise: bool = False
    AutoGradForce: bool = False
    supervise_force_from_head_when_autograd: bool = False
    NoisePredForce: bool = False
    seq_only: bool = False
    freeze_backbone: bool = False
    hard_dist_loss_raito: float = 1.0
    use_hard_dist_loss: bool = False
    if_total_energy: bool = False
    group_optimizer: bool = False
    group_lr_ratio: float = 1.0

    # used in force and noise heads
    num_force_and_noise_head_layers: int = 2

    # used for finetuning and diffusion sampling
    psm_validation_mode: bool = False
    sample_in_validation: bool = False
    num_sampling_time: int = 1
    sampled_structure_output_path: Optional[str] = None
    psm_finetune_mode: bool = False
    psm_sample_structure_in_finetune: bool = False
    psm_finetune_reset_head: bool = False
    psm_finetune_noise_mode: str = "diffusion"
    psm_finetune_valid_noise_mode: str = "diffusion"
    psm_finetune_skip_ori_head: bool = False
    only_use_rotary_embedding_for_protein: bool = False
    psm_validate_for_train_set: bool = False
    # only for dpm solver
    algorithm_type: str = "dpmsolver++"
    solver_order: int = 2
    solver_type: str = "midpoint"

    # for matbench finetuning
    psm_matbench_task_name: str = ""
    psm_matbench_fold_id: int = 0

    # dali pipeline
    use_dali_pipeline: bool = False

    # for structure relaxation
    relax_after_sampling_structure: bool = False
    structure_relax_step_size: float = 0.01
    use_autograd_force_for_relaxation_and_md: bool = False
    relax_ase_steps: int = 8000
    relax_initial_cell_matrix: str = "1,1,1"
    relax_lower_deformation: int = 0
    relax_upper_deformation: int = 0
    relax_deformation_step: int = 5
    relax_fmax: float = 0.01

    #
    max_residue_num: int = 384

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))
        self.relax_initial_cell_matrix = [
            int(i) for i in self.relax_initial_cell_matrix.split(",")
        ]

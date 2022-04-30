import argparse
import os
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from train import retrieve_clusters
from utils import load_snp, BuildIntegratedScore, BuildSumScore
from scipy import stats
import pickle



def main():

    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of NETTAG')
    parser.add_argument('--rand_seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--bin_num', type=int, default=1000,
                      help='sort nodes into bins according to degrees')
    parser.add_argument('--dirnet', type=str, default='../data/ppi_remove_self_loop.txt',
                        help='directory of network')
    parser.add_argument('--preprocess', type=str, default = None,
                        help='feature preprocessing: None, normalize, standardscaler')
    parser.add_argument('--adjpow', type=int, default=2,
                        help='element-wise power of adjacency matrix')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--K', type=int, default=1200,
                        help='cluster_number')
    parser.add_argument('--batch_size', type=int, default=14000,
                        help='input batch size for subsampling')
    parser.add_argument('--batch_norm', type=bool, default=True,
                        help='whether or not perform batch normalization')
    parser.add_argument('--epochs', type=int, default=50000,
                        help='number of epochs for iteration')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience tolerance for early stopping')
    parser.add_argument('--val_step', type=int, default=500,
                        help='validation step to evaluate loss')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='minimum learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='decrease coefficient of learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout probability')
    parser.add_argument('--hidden_size', nargs='+', help='hidden size of gnn, default = [2048, 1024]',
                        required=True) # python main.py --hidden_size 2048 1024
    parser.add_argument('--dirresult', type=str, default="../result_subsampling/",
                        help='output file')
    args = parser.parse_args()

    if not os.path.isdir(args.dirresult):
      os.mkdir(args.dirresult)

    
    '''
    Step 1: retrieve clusters of PPIs
    '''

    cluster_results, save_file_name = retrieve_clusters(args)

    # with open('../result_subsampling/NAdam_A_poly2_wd0.0001_dropout0.0_lr_0.0001_K1200_bs14000_patience5_lrmin1e-06_max-epochs50000_preprocessNone_hidden-size_2048_1024_cluster_results.pickle', 'rb') as handle:
    #   cluster_results = pickle.load(handle)
    # handle.close()


    '''
    Step 2: get predicted scores
    '''

    CpG_island = load_snp('../data/CpG_island_AD_mapped_snps_entrez_id_v3.txt', header = True)
    CTCF = load_snp('../data/CTCF_AD_mapped_snps_entrez_id_v3.txt', header = True)
    enhancer = load_snp('../data/enhancer_AD_mapped_snps_entrez_id_v3.txt', header = True)
    eQTL = load_snp('../data/eQTL_AD_mapped_snps_entrez_id_v3.txt', header = True)
    histone = load_snp('../data/histone_AD_mapped_snps_entrez_id_v3.txt', header = True)
    open_chromatin = load_snp('../data/open_chromatin_AD_mapped_snps_entrez_id_v3.txt', header = True)
    promoter = load_snp('../data/promoter_AD_mapped_snps_entrez_id_v3.txt', header = True)
    pfr = load_snp('../data/pfr_AD_mapped_snps_entrez_id_v3.txt', header = True)
    tf = load_snp('../data/TF_AD_mapped_snps_entrez_id_v3.txt', header = True)

    snp_input = dict()
    snp_input['CpG_island'] = CpG_island
    snp_input['CTCF'] = CTCF
    snp_input['enhancer'] = enhancer
    snp_input['eQTL'] = eQTL
    snp_input['histone'] = histone
    snp_input['open_chromatin'] = open_chromatin
    snp_input['promoter'] = promoter
    snp_input['pfr'] = pfr
    snp_input['tf'] = tf

    all_genes, feature_score = BuildSumScore(cluster_results, snp_input)
    all_genes, integrated_score = BuildIntegratedScore(all_genes, feature_score)

    
    f_out = open(args.dirresult + save_file_name + "_predicted_score.txt", "a")
    f_out.write('node_id' + "\t" + 'score' + "\n")

    for gene_id, score in zip(all_genes, integrated_score):
      f_out.write(gene_id + "\t" + score + "\n")

    f_out.close()


    '''
    Step 3: generate likely AD-associated genes
    '''

    signif_gene_id = []
    signif_score = []
    for gene_id, score in zip(all_genes, integrated_score):
      if score > 0:
        signif_gene_id.append(gene_id)
        signif_score.append(score)

    sig_score_mean = np.array(signif_score).mean()
    sig_score_std = np.array(signif_score).std()

    alzRG_id = []
    alzRG_score = []
    for gene_id, score in zip(signif_gene_id, signif_score):
      if (score - sig_score_mean) / sig_score_std >= stats.norm.ppf(0.99):
        alzRG_id.append(gene_id)
        alzRG_score.append(score)


    
    f_out = open(args.dirresult + save_file_name + "_predicted_alzRGs.txt", "a")
    f_out.write('node_id' + "\t" + 'score' + "\n")
    for gene_id, score in zip(alzRG_id, alzRG_score):
      f_out.write(gene_id + "\t" + score + "\n")
    f_out.close()

    


if __name__ == '__main__':
    main()















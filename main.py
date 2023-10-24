import argparse

import pandas as pd

from data.datautils import extract_feature, load_datasets, train_class_distribution, data_config
from data.oxfordpet_feature import oxfordpet_dataloader
from data.uecfood100_feature import uecfood100_datalaoder
from model import SoftmaxLinear, ReplayMemory
from model.InverseNetwork import InverseNetworkQuadChannel, inverseNetworkTrainingQuadChannel
from model.QCCPN import QCCPN, QCCPNTraining, classIncrementalQCCPNTraining
from model.ReplayMemory import MemoryScheme, MemorySchemeQCCPN
from model.SoftmaxLinear import LinearSoftmax, linearSoftmaxTraining
from myutils import make_dir, get_device
import torch
import numpy as np
import random
import os

parser = argparse.ArgumentParser(description='Continual Open-Set Recognition')
parser.add_argument('exp_type', default="main", type=str, help='type of the experimentation')
parser.add_argument('--exp_name', default=None, type=str, help='id for the experiment.')
parser.add_argument('--dataset', default="oxfordpet_c100", type=str, help='name of the dataset')
parser.add_argument('--backbone', default="mobilenet_v3_large", type=str, help='options: [mobilenet_v3_large, mobilenet_v3_small, wide_resnet50_2, vit_b_16]')
parser.add_argument('--datapath', default=None, type=str, help='the path of the data')
parser.add_argument('--batch_size', default=32, type=int, help='batch size for the dataloader')
parser.add_argument('--model', default='qccpn', type=str, help='choose between qccpn and linear_softmax')
parser.add_argument('--feature_dim', default=1024, type=int, help='feature or latent dimention of the model')
parser.add_argument('--n_epoch', default=50, type=int, help='number of epoch for training')
parser.add_argument('--seed', default=1, type=int, help='seed random')
parser.add_argument('--trial', default=5, type=int, help='number of trial for running the model')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma parameter for QCCPN')
parser.add_argument('--contrastive', default=100.0, type=float, help='contrastive prototype parameter for QCCPN')

args = parser.parse_args()


def extract_feature_main(dataset, backbone, permute_only=False):
    ROOT = args.datapath

    if dataset == 'oxfordpet':
        trainval_dl, test_dl = oxfordpet_dataloader(ROOT, batch_size=args.batch_size)
        make_dir(f"data/raw_latent/{dataset}/train_{backbone}.npy")
        extract_feature(backbone, trainval_dl, out_path=f"data/raw_latent/{dataset}/trainval_{backbone}.npy")
        extract_feature(backbone, test_dl, out_path=f"data/raw_latent/{dataset}/test_{backbone}.npy")

    elif dataset == 'uecfood100':
        trainval_dl, test_dl = uecfood100_datalaoder(ROOT, batch_size=args.batch_size)
        make_dir(f"data/raw_latent/{dataset}/train_{backbone}.npy")
        extract_feature(backbone, trainval_dl, out_path=f"data/raw_latent/{dataset}/trainval_{backbone}.npy")
        extract_feature(backbone, test_dl, out_path=f"data/raw_latent/{dataset}/test_{backbone}.npy")

    else:
        raise Exception("dataset not implemented yet")


if __name__ == "__main__":
    # Adjustint the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    def save_pdf_rejection(logdf, title, no_auroc=False):
        plt.clf()
        logdf['test accuracy'] = logdf['test accuracy'] / 100
        if no_auroc:
            logdf = logdf.melt(id_vars=['task id'], value_vars=['test accuracy', 'baccu'])
        else:
            logdf = logdf.melt(id_vars=['task id'], value_vars=['test accuracy', 'auroc', 'baccu'])

        logdf = logdf.rename({'value': 'score', 'variable': 'metrics', 'task id': 'task'}, axis=1)
        sns.lineplot(data=logdf, x='task', y='score', hue='metrics', markers=True, style='metrics').set(title=title)
        path = os.path.join("_result", "figure", "rejection_evidence", f"{title.replace('-', '_')}.pdf")
        make_dir(path)
        plt.savefig(path)

    if args.exp_type == "reject_forget_evidence":
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('Producing the evidence of the forgetting in unknown class rejection')
        with_degree = True if args.model == 'qccpn' else False


        if args.model == 'linear_softmax':

            list_c = list(range(data_config[args.dataset]['n_class']))
            n_known = data_config[args.dataset]['n_known']
            n_unknown = data_config[args.dataset]['n_unknown']
            for i in range(1, 6):

                if i >=2 :
                    random.shuffle(list_c)

                known_class = list_c[:n_known]
                unknown_class = list_c[n_known:]

                tdl_train, tdl_test, n_input, n_output, dl_u, lat_train, lbl_train = load_datasets(args.dataset, args.backbone,
                                                                                                   verbose=1,
                                                                                                   with_degree=with_degree,
                                                                                                   validation=False,
                                                                                                   batch_size=args.batch_size,
                                                                                                   non_continual=True,
                                                                                                   known_class=known_class,
                                                                                                   unknown_class=unknown_class)
                # Non Incremental Softmax
                model = LinearSoftmax(base_feature=args.feature_dim, input_feature=n_input, n_class=n_output)
                memory = None
                logdf = linearSoftmaxTraining(device, model, '_ckpt/', f'_result/{args.dataset}/P{i}_softmax_isolated.csv',
                                              tdl_train, tdl_test, dl_u, n_epoch=args.n_epoch, feature_memory=memory,
                                              trial=args.trial, initial_seed=args.seed)

                save_pdf_rejection(logdf, f'P{i} Softmax-Isolated Learning ({args.dataset})')

                tdl_train, tdl_test, n_input, n_output, dl_u, lat_train, lbl_train = load_datasets(args.dataset,
                                                                                                   args.backbone, verbose=1,
                                                                                                   with_degree=with_degree,
                                                                                                   validation=False,
                                                                                                   batch_size=args.batch_size,
                                                                                                   non_continual=False,
                                                                                                   known_class=known_class,
                                                                                                   unknown_class=unknown_class
                                                                                                   )
                # Softmax-Naive
                model = LinearSoftmax(base_feature=args.feature_dim, input_feature=n_input, n_class=n_output)
                memory = None
                logdf = linearSoftmaxTraining(device, model, '_ckpt/', f'_result/{args.dataset}/P{i}_softmax_naive.csv', tdl_train, tdl_test, dl_u, n_epoch=args.n_epoch, feature_memory=memory,
                                              trial=args.trial, initial_seed=args.seed)
                save_pdf_rejection(logdf, f'P{i} Softmax-Naive ({args.dataset})')

                # Softmax-Replay
                model = LinearSoftmax(base_feature=args.feature_dim, input_feature=n_input, n_class=n_output)
                memory = MemoryScheme(device, lat_train, lbl_train, per_class=50)
                logdf = linearSoftmaxTraining(device, model, '_ckpt/', f'_result/{args.dataset}/P{i}_softmax_replay.csv', tdl_train, tdl_test,
                                              dl_u, n_epoch=args.n_epoch, feature_memory=memory,
                                              trial=args.trial, initial_seed=args.seed)
                save_pdf_rejection(logdf, f'P{i} Softmax-Replay ({args.dataset})')

        elif args.model == 'qccpn':
            # Non Incremental QCCPN
            print("Feature DIM each channel for QCCPN:", int(args.feature_dim/4))
            print("Contrastive prototype parameter for QCCPN:", args.contrastive)

            list_c = list(range(data_config[args.dataset]['n_class']))
            n_known = data_config[args.dataset]['n_known']
            n_unknown = data_config[args.dataset]['n_unknown']
            for i in range(1, 6):

                if i >= 2:
                    random.shuffle(list_c)

                known_class = list_c[:n_known]
                unknown_class = list_c[n_known:]

                tdl_train, tdl_test, n_input, n_output, dl_u, lat_train, lat90_train, lat180_trian, lat270_train, lbl_train = load_datasets(args.dataset,
                                                                                                   args.backbone, verbose=1,
                                                                                                   with_degree=with_degree,
                                                                                                   validation=False,
                                                                                                   batch_size=args.batch_size,
                                                                                                   non_continual=True,
                                                                                                    known_class=known_class,
                                                                                                    unknown_class=unknown_class)

                model = QCCPN(feature_dim=int(args.feature_dim/4), input_feature=n_input, n_class=n_output, gamma=args.gamma)
                memory = None
                logdf = QCCPNTraining(device, model, '_ckpt/', f'_result/{args.dataset}/P{i}_qccpn_isolated_c{int(args.contrastive)}.csv',
                                              tdl_train, tdl_test, dl_u, n_epoch=args.n_epoch, feature_memory=memory,
                                      gamma=args.gamma, contrastive=args.contrastive, trial=args.trial, initial_seed=args.seed)
                save_pdf_rejection(logdf, f'P{i} QCCPN-Isolated Learning C={int(args.contrastive)} ({args.dataset})')
                # ####################
                # QCCPN-Naive
                tdl_train, tdl_test, n_input, n_output, dl_u, lat_train, lat90_train, lat180_trian, lat270_train, lbl_train = load_datasets(args.dataset,
                                                                                                   args.backbone, verbose=1,
                                                                                                   with_degree=with_degree,
                                                                                                   validation=False,
                                                                                                   batch_size=args.batch_size,
                                                                                                   non_continual=False,
                                                                                                    known_class=known_class,
                                                                                                    unknown_class=unknown_class
                                                                                                    )

                model = QCCPN(feature_dim=int(args.feature_dim / 4), input_feature=n_input, n_class=n_output,
                              gamma=args.gamma)
                memory = None
                logdf = QCCPNTraining(device, model, '_ckpt/', f'_result/{args.dataset}/P{i}_qccpn_naive_c{int(args.contrastive)}.csv',
                                      tdl_train, tdl_test, dl_u, n_epoch=args.n_epoch, feature_memory=memory,
                                      gamma=args.gamma, contrastive=args.contrastive, trial=args.trial, initial_seed=args.seed)
                save_pdf_rejection(logdf, f'P{i} QCCPN-Naive C={int(args.contrastive)} ({args.dataset})')
                ################
                # QCCPN-Replay
                tdl_train, tdl_test, n_input, n_output, dl_u, lat_train, lat90_train, lat180_trian, lat270_train, lbl_train = load_datasets(args.dataset,
                                                                                                   args.backbone, verbose=1,
                                                                                                   with_degree=with_degree,
                                                                                                   validation=False,
                                                                                                   batch_size=args.batch_size,
                                                                                                   non_continual=False,
                                                                                                    known_class=known_class,
                                                                                                    unknown_class=unknown_class
                                                                                                    )

                model = QCCPN(feature_dim=int(args.feature_dim / 4), input_feature=n_input, n_class=n_output,
                              gamma=args.gamma)
                memory = MemorySchemeQCCPN(device, lat_train, lat90_train, lat180_trian, lat270_train, lbl_train, per_class=50)
                logdf = QCCPNTraining(device, model, '_ckpt/', f'_result/{args.dataset}/P{i}_qccpn_replay_c{int(args.contrastive)}.csv',
                                      tdl_train, tdl_test, dl_u, n_epoch=args.n_epoch, feature_memory=memory,
                                      gamma=args.gamma, contrastive=args.contrastive, trial=args.trial, initial_seed=args.seed)
                save_pdf_rejection(logdf, f'P{i} QCCPN-Replay C={int(args.contrastive)} ({args.dataset})')

        else:
            raise Exception(f"model {args.model} is not available")

    elif args.exp_type == "reject_forget_evidence_per_metric":
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns

        def save_pdf_rejection_per_metric(df_concat, title, y='test accuracy', prefix=""):
            plt.clf()
            df_concat['test accuracy'] = df_concat['test accuracy'] / 100
            df_concat = df_concat.rename({"task id": "task"}, axis=1)
            sns.lineplot(data=df_concat, x='task', y=y, hue='method', markers=True, style='method').set(
                title=title)
            path = os.path.join("_result", "figure", "rejection_evidence_per_metric", args.dataset, f"{prefix}_{title.replace('-', '_')}.pdf")
            make_dir(path)
            plt.savefig(path)

        for i in range(1, 6):
            # df_qccpn_isolated = pd.read_csv(os.path.join("_result", args.dataset, f'P{i}_qccpn_isolated_c{int(args.contrastive)}.csv'))
            # df_qccpn_isolated['method'] = 'QCCPN-Isolated Learning'
            df_qccpn_naive = pd.read_csv(os.path.join("_result", args.dataset, f'P{i}_qccpn_naive_c{int(args.contrastive)}.csv'))
            df_qccpn_naive['method'] = "QCCPN-Naive"
            df_qccpn_replay = pd.read_csv(os.path.join("_result", args.dataset, f'P{i}_qccpn_replay_c{int(args.contrastive)}.csv'))
            df_qccpn_replay['method'] = "QCCPN-Replay"

            # df_softmax_isolated = pd.read_csv(os.path.join("_result", args.dataset, f'P{i}_softmax_isolated.csv'))
            # df_softmax_isolated['method'] = "Softmax-Isolated Learning"
            df_softmax_naive = pd.read_csv(os.path.join("_result", args.dataset, f'P{i}_softmax_naive.csv'))
            df_softmax_naive['method'] = "Softmax-Naive"
            df_softmax_replay = pd.read_csv(os.path.join("_result", args.dataset, f'P{i}_softmax_replay.csv'))
            df_softmax_replay ['method'] = "Softmax-Replay"

            df_ci_qccpn = pd.read_csv(
                os.path.join("_result", args.dataset, f'P{i}_ci_qccpn_c{int(args.contrastive)}.csv'))
            df_ci_qccpn['method'] = 'CI-QCCPN-Replay'

            # df_concat = pd.concat([df_qccpn_isolated, df_qccpn_naive, df_qccpn_replay, df_softmax_isolated, df_softmax_naive, df_softmax_replay, df_ci_qccpn], ignore_index=True)
            df_concat = pd.concat([df_qccpn_naive, df_qccpn_replay, df_softmax_naive, df_softmax_replay, df_ci_qccpn], ignore_index=True)

            save_pdf_rejection_per_metric(df_concat, f"P{i} Test Accuracy", y='test accuracy', prefix=f"P{i}")
            save_pdf_rejection_per_metric(df_concat, f"P{i} AUROC", y='auroc', prefix=f"P{i}")
            save_pdf_rejection_per_metric(df_concat, f"P{i} BACCU", y='baccu', prefix=f"P{i}")

    elif args.exp_type == 'inverse_network':
        import json
        device = get_device()

        tdl_train, tdl_test, n_input, n_output, dl_u, lat_train, lat90_train, lat180_trian, lat270_train, lbl_train = load_datasets(
            args.dataset,
            args.backbone, verbose=1,
            with_degree=True,
            validation=False,
            batch_size=args.batch_size,
            non_continual=False,
            one_task=True)

        modelQC = QCCPN(feature_dim=int(args.feature_dim / 4), input_feature=n_input, n_class=n_output,
                        gamma=args.gamma)

        save_path = f'_ckpt/qccpn_inverse_network/qccpn_{args.dataset}.pth'
        logdf, modelQC = QCCPNTraining(device, modelQC, save_path,
                                       None,
                                       tdl_train, tdl_test, dl_u, n_epoch=args.n_epoch, feature_memory=None,
                                       gamma=args.gamma, contrastive=args.contrastive, trial=1,
                                       initial_seed=args.seed, return_model=True, save_last_model=True)

        # with open('load_path.json') as f:
        #     data = json.load(f)
        #     print(data)
        #
        # with open('load_path.json', 'w') as f:
        #     data[f'qccpn_inverse_network_{args.dataset}'] = save_path
        #     json.dump(data, f)

        inet = InverseNetworkQuadChannel(int(args.feature_dim/4), backbone_dim=960)
        mse1 = inverseNetworkTrainingQuadChannel(device, inet, modelQC, None, None, tdl_train[0], dl_u, feature_memory=None, feature_mixup=False,
                                          n_epoch=args.n_epoch, initial_seed=args.seed)
        # mse2 = inverseNetworkTrainingQuadChannel(device, inet, modelQC, None, None, tdl_train[0], dl_u, feature_memory=None,
        #                                   feature_mixup=True,
        #                                   n_epoch=args.n_epoch, initial_seed=args.seed)
        # df = pd.DataFrame({"mse": [mse1], "mse_mixup": [mse2]})
        df = pd.DataFrame({"mse": [mse1]})
        dfpath = f"_result/{args.dataset}/inverse_network.csv"
        make_dir(dfpath)
        df.to_csv(dfpath)
    elif args.exp_type == "ci_qccpn":
        # TODO save mechanism
        import matplotlib.pyplot as plt
        import seaborn as sns
        with_degree = True
        device = get_device()

        list_c = list(range(data_config[args.dataset]['n_class']))
        n_known = data_config[args.dataset]['n_known']
        n_unknown = data_config[args.dataset]['n_unknown']
        for i in range(1, 6):

            if i >= 2:
                random.shuffle(list_c)

            known_class = list_c[:n_known]
            unknown_class = list_c[n_known:]
            tdl_train, tdl_test, n_input, n_output, dl_u, lat_train, lat90_train, lat180_trian, lat270_train, lbl_train = load_datasets(
                args.dataset,
                args.backbone, verbose=1,
                with_degree=with_degree,
                validation=False,
                batch_size=args.batch_size,
                non_continual=False,
            known_class=known_class,
            unknown_class=unknown_class)

            model = QCCPN(feature_dim=int(args.feature_dim / 4), input_feature=n_input, n_class=n_output, gamma=args.gamma)
            memory = MemorySchemeQCCPN(device, lat_train, lat90_train, lat180_trian, lat270_train, lbl_train, per_class=50)
            print("backbone dim:", n_input)
            logdf = classIncrementalQCCPNTraining(device, model, '_ckpt/',
                                  f'_result/{args.dataset}/P{i}_ci_qccpn_c{int(args.contrastive)}.csv',
                                  tdl_train, tdl_test, dl_u, n_epoch=args.n_epoch, feature_memory=memory,
                                  gamma=args.gamma, contrastive=args.contrastive, trial=args.trial, initial_seed=args.seed, feature_dim=args.feature_dim, backbone_dim=n_input)
            save_pdf_rejection(logdf, f'P{i} CI-QCCPN C={int(args.contrastive)} ({args.dataset})')

    elif args.exp_type == 'reject_performance_qccpn':
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns


        def save_pdf_rejection_per_metric(df_concat, title, y='test accuracy', prefix=""):
            plt.clf()
            df_concat['test accuracy'] = df_concat['test accuracy'] / 100
            df_concat = df_concat.rename({"task id": "task"}, axis=1)
            sns.lineplot(data=df_concat, x='task', y=y, hue='method', markers=True, style='method').set(
                title=title)
            path = os.path.join("_result", "figure", "reject_performance_qccpn", args.dataset,
                                f"{prefix}_{title.replace('-', '_')}.pdf")
            make_dir(path)
            plt.savefig(path)

        for i in range(1, 6):
            df_ci_qccpn = pd.read_csv(
                os.path.join("_result", args.dataset, f'P{i}_ci_qccpn_c{int(args.contrastive)}.csv'))
            df_ci_qccpn['method'] = 'CI-QCCPN-Replay'
            df_qccpn_replay = pd.read_csv(
                os.path.join("_result", args.dataset, f'P{i}_qccpn_replay_c{int(args.contrastive)}.csv'))
            df_qccpn_replay['method'] = "QCCPN-Replay"

            df_concat = pd.concat(
                [df_ci_qccpn, df_qccpn_replay], ignore_index=True)

            save_pdf_rejection_per_metric(df_concat, "Test Accuracy", y='test accuracy', prefix=f"P{i}")
            save_pdf_rejection_per_metric(df_concat, "AUROC", y='auroc', prefix=f"P{i}")
            save_pdf_rejection_per_metric(df_concat, "BACCU", y='baccu', prefix=f"P{i}")

    elif args.exp_type == 'train_class_distribution':
        # Oxfordpet 50
        train_class_distribution(args.dataset)
    elif args.exp_type == "extract_feature":
        print(f"Extracting the feature of {args.dataset} through {args.backbone}")
        extract_feature_main(args.dataset, args.backbone)
    elif args.exp_type == "table_comparison":
        data = {"Dataset": [], "Softmax-Naive": [], "Softmax-Replay": [], "QCCPN-Naive": [], "QCCPN-Replay": [], "CI-QCCPN": []}
        for i in range(1, 4):
            m1 = pd.read_csv(os.path.join("_result", "oxfordpet", f"P{i}_softmax_naive.csv"))
            m2 = pd.read_csv(os.path.join("_result", "oxfordpet", f"P{i}_softmax_replay.csv"))
            m3 = pd.read_csv(os.path.join("_result", "oxfordpet", f"P{i}_qccpn_naive_c50.csv"))
            m4 = pd.read_csv(os.path.join("_result", "oxfordpet", f"P{i}_qccpn_replay_c50.csv"))
            m5 = pd.read_csv(os.path.join("_result", "oxfordpet", f"P{i}_ci_qccpn_c50.csv"))

            m1['test accuracy'] = m1['test accuracy'] / 100
            m2['test accuracy'] = m2['test accuracy'] / 100
            m3['test accuracy'] = m3['test accuracy'] / 100
            m4['test accuracy'] = m4['test accuracy'] / 100
            m5['test accuracy'] = m5['test accuracy'] / 100

            data['Dataset'].append(f"Oxfordpet P({i})")
            data['Softmax-Naive'].append(f"{np.round(m1['test accuracy'].mean(), 3)} ± {np.round(m1['test accuracy'].std(), 3)} ({np.round(m1['auroc'].mean(), 3)} ± {np.round(m1['auroc'].std(), 3)})")
            data['Softmax-Replay'].append(f"{np.round(m2['test accuracy'].mean(), 3)} ± {np.round(m2['test accuracy'].std(), 3)} ({np.round(m2['auroc'].mean(), 3)} ± {np.round(m2['auroc'].std(), 3)})")
            data['QCCPN-Naive'].append(f"{np.round(m3['test accuracy'].mean(), 3)} ± {np.round(m3['test accuracy'].std(), 3)} ({np.round(m3['auroc'].mean(), 3)} ± {np.round(m3['auroc'].std(), 3)})")
            data['QCCPN-Replay'].append(f"{np.round(m4['test accuracy'].mean(), 3)} ± {np.round(m4['test accuracy'].std(), 3)} ({np.round(m4['auroc'].mean(), 3)} ± {np.round(m4['auroc'].std(), 3)})")
            data['CI-QCCPN'].append(f"{np.round(m5['test accuracy'].mean(), 3)} ± {np.round(m5['test accuracy'].std(), 3)} ({np.round(m5['auroc'].mean(), 3)} ± {np.round(m5['auroc'].std(), 3)})")

            if i <= 1:
                cm1 = m1
                cm2 = m2
                cm3 = m3
                cm4 = m4
                cm5 = m5
            else:
                cm1 = pd.concat([cm1, m1])
                cm2 = pd.concat([cm2, m2])
                cm3 = pd.concat([cm3, m3])
                cm4 = pd.concat([cm4, m4])
                cm5 = pd.concat([cm5, m5])

        for i in range(1, 4):
            m1 = pd.read_csv(os.path.join("_result", "uecfood100", f"P{i}_softmax_naive.csv"))
            m2 = pd.read_csv(os.path.join("_result", "uecfood100", f"P{i}_softmax_replay.csv"))
            m3 = pd.read_csv(os.path.join("_result", "uecfood100", f"P{i}_qccpn_naive_c50.csv"))
            m4 = pd.read_csv(os.path.join("_result", "uecfood100", f"P{i}_qccpn_replay_c50.csv"))
            m5 = pd.read_csv(os.path.join("_result", "uecfood100", f"P{i}_ci_qccpn_c50.csv"))

            m1['test accuracy'] = m1['test accuracy']/100
            m2['test accuracy'] = m2['test accuracy']/100
            m3['test accuracy'] = m3['test accuracy']/100
            m4['test accuracy'] = m4['test accuracy']/100
            m5['test accuracy'] = m5['test accuracy']/100

            data['Dataset'].append(f"UECFOOD100 P({i})")
            data['Softmax-Naive'].append(f"{np.round(m1['test accuracy'].mean(), 3)} ± {np.round(m1['test accuracy'].std(), 3)} ({np.round(m1['auroc'].mean(), 3)} ± {np.round(m1['auroc'].std(), 3)})")
            data['Softmax-Replay'].append(f"{np.round(m2['test accuracy'].mean(), 3)} ± {np.round(m2['test accuracy'].std(), 3)} ({np.round(m2['auroc'].mean(), 3)} ± {np.round(m2['auroc'].std(), 3)})")
            data['QCCPN-Naive'].append(f"{np.round(m3['test accuracy'].mean(), 3)} ± {np.round(m3['test accuracy'].std(), 3)} ({np.round(m3['auroc'].mean(), 3)} ± {np.round(m3['auroc'].std(), 3)})")
            data['QCCPN-Replay'].append(f"{np.round(m4['test accuracy'].mean(), 3)} ± {np.round(m4['test accuracy'].std(), 3)} ({np.round(m4['auroc'].mean(), 3)} ± {np.round(m4['auroc'].std(), 3)})")
            data['CI-QCCPN'].append(f"{np.round(m5['test accuracy'].mean(), 3)} ± {np.round(m5['test accuracy'].std(), 3)} ({np.round(m5['auroc'].mean(), 3)} ± {np.round(m5['auroc'].std(), 3)})")

            cm1 = pd.concat([cm1, m1])
            cm2 = pd.concat([cm2, m2])
            cm3 = pd.concat([cm3, m3])
            cm4 = pd.concat([cm4, m4])
            cm5 = pd.concat([cm5, m5])

        data = pd.DataFrame(data)
        data.to_csv(f"_result/table_comparison.csv")
        data2 = {
            'Softmax-Naive': [f"{np.round(cm1['test accuracy'].mean(), 3)} ± {np.round(cm1['test accuracy'].std(), 3)} ({np.round(cm1['auroc'].mean(), 3)} ± {np.round(cm1['auroc'].std(), 3)})"],
            'Softmax-Replay': [f"{np.round(cm2['test accuracy'].mean(), 3)} ± {np.round(cm2['test accuracy'].std(), 3)} ({np.round(cm2['auroc'].mean(), 3)} ± {np.round(cm2['auroc'].std(), 3)})"],
            'QCCPN-Naive': [f"{np.round(cm3['test accuracy'].mean(), 3)} ± {np.round(cm3['test accuracy'].std(), 3)} ({np.round(cm3['auroc'].mean(), 3)} ± {np.round(cm3['auroc'].std(), 3)})"],
            'QCCPN-Replay': [f"{np.round(cm4['test accuracy'].mean(), 3)} ± {np.round(cm4['test accuracy'].std(), 3)} ({np.round(cm4['auroc'].mean(), 3)} ± {np.round(cm4['auroc'].std(), 3)})"],
            'CI-QCCPN': [f"{np.round(cm5['test accuracy'].mean(), 3)} ± {np.round(cm5['test accuracy'].std(), 3)} ({np.round(cm5['auroc'].mean(), 3)} ± {np.round(cm5['auroc'].std(), 3)})"]
        }
        pd.DataFrame(data2).to_csv(f"_result/summary_table_comparison.csv")

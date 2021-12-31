import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib
import itertools
matplotlib.use('Agg')
from pathlib import Path
from lib import GenerationR, train_test_gpr, validate_gpr
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser(description='sMRI-DP')
parser.add_argument('-df', '--dataframe', type=Path)
parser.add_argument('-tr', '--train-df', type=Path)
parser.add_argument('-te', '--test-df', type=Path)
parser.add_argument(
    '-ulw', '--use-last-wave', action='store_true', default=False)
parser.add_argument('-ep', '--num-epochs', type=int, default=100)
parser.add_argument('--random-state', '-rs', type=int, default=42)
args = parser.parse_args()

if __name__ == '__main__':
    results_p = Path('./results')
    results_p.mkdir(parents=True, exist_ok=True)
    ds = GenerationR(df_path=args.dataframe,
                     train_df=args.train_df,
                     test_df=args.test_df,
                     random_state=args.random_state,
                     use_last_wave=args.use_last_wave)
    min_age, max_age = ds.min_max
    differential_num_steps = 1000
    dt = 1/(differential_num_steps - 1)
    diff_range = torch.linspace(0, 1, differential_num_steps)
    brain_measures = ['volume', 'thickness']
    models = ['model2']
    brain_measures = ['volume']
    name_map = {'volume': 'vol', 'thickness': 'thickavg'}
    variables_not_to_correct = []
    means = ['ZeroMean']
    kernels = ['LinearKernel', 'MaternKernel', 'RBFKernel', 'RQKernel']
    model_params = list(itertools.product(means, kernels))
    pvalues_dict = {model: {'int_symptoms': None,
                            'ext_symptoms': None,
                            'dp_symptoms': None} for model in models}
    tvalues_dict = {model: {'int_symptoms': None,
                            'ext_symptoms': None,
                            'dp_symptoms': None} for model in models}
    pvalues_dict_abs = {
        model: {'int_symptoms': None,
                'ext_symptoms': None,
                'dp_symptoms': None} for model in models}
    tvalues_dict_abs = {
        model: {'int_symptoms': None,
                'ext_symptoms': None,
                'dp_symptoms': None} for model in models}
    kernel_dict_full = {model: None for model in models}
    pred_dict_full = {model: None for model in models}
    for model in models:
        pvalues_dict[model]['int_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        pvalues_dict[model]['ext_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        pvalues_dict[model]['dp_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        tvalues_dict[model]['int_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        tvalues_dict[model]['ext_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        tvalues_dict[model]['dp_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        pvalues_dict_abs[model]['int_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        pvalues_dict_abs[model]['ext_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        pvalues_dict_abs[model]['dp_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        tvalues_dict_abs[model]['int_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        tvalues_dict_abs[model]['ext_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        tvalues_dict_abs[model]['dp_symptoms'] = {
            brain_measure: None for brain_measure in brain_measures}
        kernel_dict_full[model] = {
            brain_measure: None for brain_measure in brain_measures}
        pred_dict_full[model] = {
            brain_measure: None for brain_measure in brain_measures}
        for brain_measure in brain_measures:
            brain_regions = ds.model_dict[model][brain_measure]
            region_names = [reg.replace(
                f'resid_{model}', '').replace(
                    f'_{name_map[brain_measure]}_', ' ').replace(
                        '_', ' ').capitalize().rstrip()
                            for reg in brain_regions]
            pvalues_dict[model]['int_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            pvalues_dict[model]['ext_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            pvalues_dict[model]['dp_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            tvalues_dict[model]['int_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            tvalues_dict[model]['ext_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            tvalues_dict[model]['dp_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            pvalues_dict_abs[model]['int_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            pvalues_dict_abs[model]['ext_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            pvalues_dict_abs[model]['dp_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            tvalues_dict_abs[model]['int_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            tvalues_dict_abs[model]['ext_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            tvalues_dict_abs[model]['dp_symptoms'][brain_measure] = {
                region_name: None for region_name in region_names}
            kernel_dict_full[model][brain_measure] = {
                region_name: None for region_name in region_names
            }
            pred_dict_full[model][brain_measure] = {
                region_name: None for region_name in region_names
            }
            regression_dict = {region: {'z_scores': [],
                                        'int_symptoms': [],
                                        'ext_symptoms': [],
                                        'dp_symptoms': []}
                               for region in region_names}
            range_dict = {region: {'range_in': [],
                                   'range_pred': []}
                          for region in region_names}
            train_dict = {region_name: {} for region_name in region_names}
            diff_df = pd.DataFrame(
                np.zeros((len(brain_regions), 4)),
                index=region_names,
                columns=['min', 'max', 'avg', 'std'])
            for (region_name, brain_region) in \
                    zip(region_names, brain_regions):
                print(f'Validating region: {region_name}')
                train_dict[region_name] = {i: {'x_train': None,
                                               'y_train': None}
                                           for i in range(len(ds.folds))}
                kernel_perf_dict = {f'{kernel}_{mean}': []
                                    for mean, kernel in model_params}
                #################################################
                # Performing validation
                #################################################
                for (i, (mean, kernel)) in enumerate(model_params):
                    key = f'{kernel}_{mean}'
                    for (k, fold) in enumerate(ds.folds):
                        if not (fold['test'].empty or fold['train'].empty
                                or fold['valid'].empty):
                            x_train = fold['train'].loc[:, 'age_mri'].values
                            x_valid = fold['valid'].loc[:, 'age_mri'].values

                            x_train = (x_train - min_age) / (max_age - min_age)
                            x_valid = (x_valid - min_age) / (max_age - min_age)

                            y_train = np.reshape(fold['train'].loc[
                                :, brain_region].values, (-1, 1))
                            y_valid = np.reshape(fold['valid'].loc[
                                :, brain_region].values, (-1, 1))

                            scaler = StandardScaler(
                                with_mean=True, with_std=True)
                            y_train = scaler.fit_transform(y_train)
                            y_valid = scaler.transform(y_valid)
                            del scaler

                            ll = validate_gpr(
                                mean, kernel,
                                x_train, x_valid,
                                y_train, y_valid,
                                num_epochs=args.num_epochs)
                            kernel_perf_dict[key].append(ll)
                best_ll = -np.inf
                for (mean, kernel) in model_params:
                    key = f'{kernel}_{mean}'
                    if np.array(kernel_perf_dict[key]).mean() > best_ll:
                        best_ll = np.array(kernel_perf_dict[key]).mean()
                        best_kernel = kernel
                        best_mean = mean
                        best_key = key
                print(f'Brain region: {region_name}, '
                      f'best kernel: {best_kernel}, '
                      f'best mean: {best_mean}')
                kernel_dict_full[model][brain_measure][region_name] = best_key
                pred_df = pd.DataFrame()

                print(f'Predicting region: {region_name}')

                ####################################################
                # Test set prediction
                ####################################################
                diff_ls = []
                for (k, fold) in enumerate(ds.folds):
                    if not (fold['test'].empty or fold['train'].empty
                            or fold['valid'].empty):
                        x_train = fold['train'].loc[:, 'age_mri'].values
                        x_valid = fold['valid'].loc[:, 'age_mri'].values
                        x_train = np.concatenate((x_train, x_valid))
                        x_test = fold['test'].loc[:, 'age_mri'].values

                        x_train = (x_train - min_age) / (max_age - min_age)
                        x_valid = (x_valid - min_age) / (max_age - min_age)
                        x_test = (x_test - min_age) / (max_age - min_age)

                        y_train = np.reshape(fold['train'].loc[
                            :, brain_region].values, (-1, 1))
                        y_valid = np.reshape(fold['valid'].loc[
                            :, brain_region].values, (-1, 1))
                        y_train = np.concatenate(
                            (y_train, y_valid))
                        y_test = np.reshape(fold['test'].loc[
                            :, brain_region].values, (-1, 1))

                        scaler = StandardScaler()
                        y_train = scaler.fit_transform(y_train)
                        y_test = scaler.transform(y_test)

                        train_dict[region_name][k]['x_train'] = x_train
                        train_dict[region_name][k]['y_train'] = y_train
                        del scaler

                        range_in, range_pred, \
                            test_pred, trained_model, \
                            trained_likelihood = train_test_gpr(
                                best_mean, best_kernel,
                                x_train, x_test,
                                y_train, y_test,
                                num_epochs=args.num_epochs)

                        with torch.no_grad():
                            diff_y = trained_likelihood(
                                trained_model(diff_range)).mean
                        diff_ls.append(diff_y)

                        range_dict[region_name]['range_in'].append(
                            range_in)
                        range_dict[region_name]['range_pred'].append(
                            range_pred)

                        int_symptoms = fold['test'].loc[
                            :, ds.model_dict[model]['int_symptoms']].values
                        ext_symptoms = fold['test'].loc[
                            :, ds.model_dict[model]['ext_symptoms']].values
                        dp_symptoms = fold['test'].loc[
                            :, ds.model_dict[model]['dp_symptoms']].values
                        z_score \
                            = (y_test.squeeze() -
                                test_pred.mean.detach().numpy()) \
                            / test_pred.variance.detach().numpy()
                        regression_dict[region_name]['z_scores'] \
                            += z_score.tolist()
                        regression_dict[region_name]['int_symptoms'] \
                            += int_symptoms.tolist()
                        regression_dict[region_name]['ext_symptoms'] \
                            += ext_symptoms.tolist()
                        regression_dict[region_name]['dp_symptoms'] \
                            += dp_symptoms.tolist()

                        temp_df = pd.DataFrame(
                            np.empty((x_test.shape[0], 8)),
                            columns=['x_test', 'y_test_mean', 'y_test_var',
                                     'y_true', 'z_scores', 'int_symptoms',
                                     'ext_symptoms', 'dp_symptoms'],
                            index=fold['test'].index)
                        temp_df['x_test'] = x_test
                        temp_df['y_test_mean'] = test_pred.mean.numpy()
                        temp_df['y_test_var'] \
                            = test_pred.variance.detach().numpy()
                        temp_df['y_true'] = y_test
                        temp_df['z_scores'] = z_score
                        temp_df['int_symptoms'] = int_symptoms
                        temp_df['ext_symptoms'] = ext_symptoms
                        temp_df['dp_symptoms'] = dp_symptoms
                        pred_df = pd.concat((pred_df, temp_df))
                fold_diff = torch.stack(diff_ls, dim=0).mean(0)
                dy_dt = (fold_diff[1:] - torch.roll(fold_diff, 1)[1:]) / dt
                diff_df.loc[region_name, :] = [dy_dt.min(), dy_dt.max(),
                                               dy_dt.mean(), dy_dt.std()]
                ############################################################################
                # Calculating p-value and rho
                ############################################################################
                pred_dict_full[model][brain_measure][region_name] = pred_df
                z_scores = np.array(regression_dict[region_name]['z_scores'])
                int_symptoms = np.array(
                    regression_dict[region_name]['int_symptoms']).squeeze()
                ext_symptoms = np.array(
                    regression_dict[region_name]['ext_symptoms']).squeeze()
                dp_symptoms = np.array(
                    regression_dict[region_name]['dp_symptoms']).squeeze()

                int_mask = np.isfinite(int_symptoms) & np.isfinite(z_scores)
                ext_mask = np.isfinite(ext_symptoms) & np.isfinite(z_scores)
                dp_mask = np.isfinite(dp_symptoms) & np.isfinite(z_scores)

                reg_int = sm.OLS(z_scores[int_mask], int_symptoms[int_mask])
                reg_ext = sm.OLS(z_scores[ext_mask], ext_symptoms[ext_mask])
                reg_dp = sm.OLS(z_scores[dp_mask], dp_symptoms[dp_mask])
                res_int = reg_int.fit()
                res_ext = reg_ext.fit()
                res_dp = reg_dp.fit()

                reg_int_abs = sm.OLS(
                    np.abs(z_scores[int_mask]), int_symptoms[int_mask])
                reg_ext_abs = sm.OLS(
                    np.abs(z_scores[ext_mask]), ext_symptoms[ext_mask])
                reg_dp_abs = sm.OLS(
                    np.abs(z_scores[dp_mask]), dp_symptoms[dp_mask])
                res_int_abs = reg_int_abs.fit()
                res_ext_abs = reg_ext_abs.fit()
                res_dp_abs = reg_dp_abs.fit()

                res_int_t, res_int_p = res_int.params[0], res_int.pvalues[0]
                res_ext_t, res_ext_p = res_ext.params[0], res_ext.pvalues[0]
                res_dp_t, res_dp_p = res_dp.params[0], res_dp.pvalues[0]

                res_int_t_abs, res_int_p_abs \
                    = res_int_abs.params[0], res_int_abs.pvalues[0]
                res_ext_t_abs, res_ext_p_abs \
                    = res_ext_abs.params[0], res_ext_abs.pvalues[0]
                res_dp_t_abs, res_dp_p_abs \
                    = res_dp_abs.params[0], res_dp_abs.pvalues[0]
                print(res_int_t, res_int_p, res_int_t_abs, res_int_p_abs)
                print(res_ext_t, res_ext_p, res_ext_t_abs, res_ext_p_abs)
                print(res_dp_t, res_dp_p, res_dp_t_abs, res_dp_p_abs)

                pvalues_dict[model][
                    'int_symptoms'][brain_measure][region_name] = res_int_p
                pvalues_dict[model][
                    'ext_symptoms'][brain_measure][region_name] = res_ext_p
                pvalues_dict[model][
                    'dp_symptoms'][brain_measure][region_name] = res_dp_p

                tvalues_dict[model][
                    'int_symptoms'][brain_measure][region_name] = res_int_t
                tvalues_dict[model][
                    'ext_symptoms'][brain_measure][region_name] = res_ext_t
                tvalues_dict[model][
                    'dp_symptoms'][brain_measure][region_name] = res_dp_t

                pvalues_dict_abs[model][
                    'int_symptoms'][brain_measure][region_name] = res_int_p_abs
                pvalues_dict_abs[model][
                    'ext_symptoms'][brain_measure][region_name] = res_ext_p_abs
                pvalues_dict_abs[model][
                    'dp_symptoms'][brain_measure][region_name] = res_dp_p_abs

                tvalues_dict_abs[model][
                    'int_symptoms'][brain_measure][region_name] = res_int_t_abs
                tvalues_dict_abs[model][
                    'ext_symptoms'][brain_measure][region_name] = res_ext_t_abs
                tvalues_dict_abs[model][
                    'dp_symptoms'][brain_measure][region_name] = res_dp_t_abs

                #######################################################
                # Plot dataset-wide results
                #######################################################
                plt_p = results_p / Path(f'{model}') \
                    / Path(f'{brain_measure}') / Path(f'{region_name}')
                if not plt_p.is_dir():
                    plt_p.mkdir(parents=True, exist_ok=True)
                x_plot = (pred_df['x_test'].values[int_mask]
                          * (max_age - min_age)) + min_age
                lower = pred_df['y_test_mean'].values[int_mask] \
                    - np.sqrt(pred_df['y_test_var'].values[int_mask])
                upper = pred_df['y_test_mean'].values[int_mask] \
                    + np.sqrt(pred_df['y_test_var'].values[int_mask])
                size_factor = 5

                x_sort_ix = x_plot.argsort()
                int_cmap = plt.get_cmap('spring')
                norm_c = (pred_df['int_symptoms'].values[int_mask]
                          - min_age) / (max_age - min_age)
                fig, ax = plt.subplots(
                    1, 1, figsize=(30, 30), sharex=True, sharey=True)
                ax.scatter(x_plot, pred_df['y_true'].values[int_mask],
                           c=norm_c,
                           cmap=int_cmap,
                           alpha=0.75,
                           s=np.power((1 + norm_c), size_factor) * 15)
                ax.plot(x_plot[x_sort_ix],
                        pred_df['y_test_mean'].values[int_mask][x_sort_ix],
                        c='b', alpha=0.25)
                ax.legend(['Observed data', 'Predicted mean'])
                ax.title.set_text(f'{region_name}')
                fig.savefig(plt_p / Path('int_age_measure.png'))
                plt.close(fig)

                fig, ax = plt.subplots(
                    1, 1, figsize=(30, 30), sharex=True, sharey=True)
                ax.scatter(x_plot, pred_df['z_scores'].values[int_mask],
                           c=norm_c,
                           cmap=int_cmap,
                           alpha=0.75,
                           s=np.power((1 + norm_c), size_factor) * 15)
                ax.set_xlabel('Age')
                ax.set_ylabel('Z score')
                ax.title.set_text(f'{region_name}')
                fig.savefig(plt_p / Path('int_age_zscores.png'))
                plt.close(fig)

                x_plot = (pred_df['x_test'].values[ext_mask]
                          * (max_age - min_age)) + min_age
                lower = pred_df['y_test_mean'].values[ext_mask] \
                    - np.sqrt(pred_df['y_test_var'].values[ext_mask])
                upper = pred_df['y_test_mean'].values[ext_mask] \
                    + np.sqrt(pred_df['y_test_var'].values[ext_mask])
                x_sort_ix = x_plot.argsort()
                ext_cmap = plt.get_cmap('summer')
                norm_c = (pred_df['ext_symptoms'].values[ext_mask]
                          - min_age) / (max_age - min_age)
                fig, ax = plt.subplots(
                    1, 1, figsize=(30, 30), sharex=True, sharey=True)
                ax.scatter(x_plot, pred_df['y_true'].values[ext_mask],
                           c=norm_c,
                           cmap=ext_cmap,
                           alpha=0.75,
                           s=np.power((1 + norm_c), size_factor) * 15)
                ax.plot(x_plot[x_sort_ix],
                        pred_df['y_test_mean'].values[ext_mask][x_sort_ix],
                        c='b',
                        alpha=0.25)
                ax.legend(['Observed data', 'Predicted mean'])
                ax.title.set_text(f'{region_name}')
                fig.savefig(plt_p / Path('ext_age_measure.png'))
                plt.close(fig)

                fig, ax = plt.subplots(
                    1, 1, figsize=(30, 30), sharex=True, sharey=True)
                ax.scatter(x_plot, pred_df['z_scores'].values[ext_mask],
                           c=norm_c,
                           cmap=ext_cmap,
                           alpha=0.75,
                           s=np.power((1 + norm_c), size_factor) * 15)
                ax.set_xlabel('Age')
                ax.set_ylabel('Z score')
                ax.title.set_text(f'{region_name}')
                fig.savefig(plt_p / Path('ext_age_zscores.png'))
                plt.close(fig)

                x_plot = (pred_df['x_test'].values[dp_mask]
                          * (max_age - min_age)) + min_age
                lower = pred_df['y_test_mean'].values[dp_mask] \
                    - np.sqrt(pred_df['y_test_var'].values[dp_mask])
                upper = pred_df['y_test_mean'].values[dp_mask] \
                    + np.sqrt(pred_df['y_test_var'].values[dp_mask])
                x_sort_ix = x_plot.argsort()
                dp_cmap = plt.get_cmap('autumn')
                norm_c = (pred_df['dp_symptoms'].values[dp_mask]
                          - min_age) / (max_age - min_age)
                fig, ax = plt.subplots(
                    1, 1, figsize=(30, 30), sharex=True, sharey=True)
                ax.scatter(x_plot, pred_df['y_true'].values[dp_mask],
                           c=norm_c,
                           cmap=dp_cmap,
                           alpha=0.75,
                           s=np.power((1 + norm_c), size_factor) * 15)
                ax.plot(x_plot[x_sort_ix],
                        pred_df['y_test_mean'].values[dp_mask][x_sort_ix],
                        c='b', alpha=0.25)
                ax.legend(['Observed data', 'Predicted mean'])
                ax.title.set_text(f'{region_name}')
                fig.savefig(plt_p / Path('ext_age_measure.png'))
                plt.close(fig)

                fig, ax = plt.subplots(
                    1, 1, figsize=(30, 30), sharex=True, sharey=True)
                ax.scatter(x_plot, pred_df['z_scores'].values[dp_mask],
                           c=norm_c,
                           cmap=dp_cmap,
                           alpha=0.75,
                           s=np.power((1 + norm_c), size_factor) * 15)
                ax.set_xlabel('Age')
                ax.set_ylabel('Z score')
                ax.title.set_text(f'{region_name}')
                fig.savefig(plt_p / Path('ext_age_zscores.png'))
                plt.close(fig)

            diff_p = Path('results') / Path(f'{model}') \
                / Path(f'{brain_measure}')
            if not diff_p.is_dir():
                diff_p.mkdir(parents=True, exist_ok=True)
            diff_df.to_csv(diff_p / Path('differential.csv'))

            for (k, fold) in enumerate(ds.folds):
                if not (fold['test'].empty or fold['train'].empty
                        or fold['valid'].empty):
                    fold_p = results_p / Path(f'fold_{k}')
                    if not fold_p.is_dir():
                        fold_p.mkdir(parents=True, exist_ok=True)
                    plt.clf()
                    rows = int(np.ceil(len(region_names) / 4))
                    fig_shape = (rows, 4)
                    fig, axs = plt.subplots(
                        *fig_shape, figsize=(30, 30), sharex=True, sharey=True)
                    for (i, (region_name, brain_region)) in enumerate(
                            zip(region_names, brain_regions)):
                        range_in = range_dict[region_name]['range_in'][k]
                        range_std = np.sqrt(range_dict[region_name][
                            'range_pred'][k].variance.detach().numpy())
                        mean_pred = range_dict[region_name][
                            'range_pred'][k].mean.detach().numpy()
                        lower = mean_pred - range_std
                        upper = mean_pred + range_std

                        y_train = np.reshape(fold['train'].loc[
                            :, brain_region].values, (-1, 1))
                        y_valid = np.reshape(fold['valid'].loc[
                            :, brain_region].values, (-1, 1))
                        y_train = np.concatenate((y_train, y_valid))
                        y_test = np.reshape(fold['test'].loc[
                            :, brain_region].values, (-1, 1))

                        scaler = StandardScaler()
                        y_train = scaler.fit_transform(y_train)
                        y_test = scaler.transform(y_test).squeeze()
                        x_plot = (train_dict[region_name][k][
                            'x_train']*(max_age - min_age)) + min_age
                        x_test = (fold['test'].loc[
                            :, 'age_mri'].values).squeeze()
                        range_in = (range_in*(max_age - min_age)) + min_age
                        axs[i % rows, i // rows].plot(range_in, mean_pred, 'b')
                        axs[i % rows, i // rows].fill_between(
                            range_in, lower, upper, alpha=0.5)
                        axs[i % rows, i // rows].scatter(
                            x_plot,
                            train_dict[region_name][k]['y_train'].ravel(),
                            color='k', alpha=0.6, s=7)
                        axs[i % rows, i // rows].scatter(
                            x_test, y_test,
                            color='r', alpha=0.6, s=7)
                        axs[i % rows, i // rows].legend(
                            ['Mean', '1 SD', 'Training data', 'Test data'])
                        axs[i % rows, i // rows].title.set_text(
                            f'{region_name}')
                    fig.savefig(fold_p / Path(f'{model} {brain_measure}.png'))
                    plt.clf()
                    plt.close(fig)

    for model in models:
        model_p = Path('private_results') / Path(f'{model}')
        if not model_p.is_dir():
            model_p.mkdir(parents=True, exist_ok=True)

        df_int_p = pd.DataFrame.from_dict(pvalues_dict[model]['int_symptoms'])
        df_ext_p = pd.DataFrame.from_dict(pvalues_dict[model]['ext_symptoms'])
        df_dp_p = pd.DataFrame.from_dict(pvalues_dict[model]['dp_symptoms'])

        df_int_t = pd.DataFrame.from_dict(tvalues_dict[model]['int_symptoms'])
        df_ext_t = pd.DataFrame.from_dict(tvalues_dict[model]['ext_symptoms'])
        df_dp_t = pd.DataFrame.from_dict(tvalues_dict[model]['dp_symptoms'])

        df_int_t_abs = pd.DataFrame.from_dict(
            tvalues_dict_abs[model]['int_symptoms'])
        df_ext_t_abs = pd.DataFrame.from_dict(
            tvalues_dict_abs[model]['ext_symptoms'])
        df_dp_t_abs = pd.DataFrame.from_dict(
            tvalues_dict_abs[model]['dp_symptoms'])

        df_int_p_abs = pd.DataFrame.from_dict(
            pvalues_dict_abs[model]['int_symptoms'])
        df_ext_p_abs = pd.DataFrame.from_dict(
            pvalues_dict_abs[model]['ext_symptoms'])
        df_dp_p_abs = pd.DataFrame.from_dict(
            pvalues_dict_abs[model]['dp_symptoms'])

        df_kernel = pd.DataFrame.from_dict(kernel_dict_full[model])
        for brain_measure in brain_measures:
            brain_measure_p = model_p / Path(f'{brain_measure}')
            if not brain_measure_p.is_dir():
                brain_measure_p.mkdir(parents=True, exist_ok=True)
            df_int = pd.DataFrame(
                np.empty((df_int_p[brain_measure].shape[0], 3)),
                columns=['Rejected', 'P-values', 'T-values'],
                index=df_int_p.index)
            df_ext = pd.DataFrame(
                np.empty((df_ext_p[brain_measure].shape[0], 3)),
                columns=['Rejected', 'P-values', 'T-values'],
                index=df_ext_p.index)
            df_dp = pd.DataFrame(
                np.empty((df_dp_p[brain_measure].shape[0], 3)),
                columns=['Rejected', 'P-values', 'T-values'],
                index=df_dp_p.index)

            df_int_abs = pd.DataFrame(
                np.empty((df_int_p_abs[brain_measure].shape[0], 3)),
                columns=['Rejected', 'P-values', 'T-values'],
                index=df_int_p.index)
            df_ext_abs = pd.DataFrame(
                np.empty((df_ext_p_abs[brain_measure].shape[0], 3)),
                columns=['Rejected', 'P-values', 'T-values'],
                index=df_ext_p.index)
            df_dp_abs = pd.DataFrame(
                np.empty((df_dp_p_abs[brain_measure].shape[0], 3)),
                columns=['Rejected', 'P-values', 'T-values'],
                index=df_dp_p.index)

            df_int['P-values'] = df_int_p[brain_measure]
            df_int['T-values'] = df_int_t[brain_measure]
            df_ext['P-values'] = df_ext_p[brain_measure]
            df_ext['T-values'] = df_ext_t[brain_measure]
            df_dp['P-values'] = df_dp_p[brain_measure]
            df_dp['T-values'] = df_dp_t[brain_measure]

            df_int_abs['P-values'] = df_int_p_abs[brain_measure]
            df_int_abs['T-values'] = df_int_t_abs[brain_measure]
            df_ext_abs['P-values'] = df_ext_p_abs[brain_measure]
            df_ext_abs['T-values'] = df_ext_t_abs[brain_measure]
            df_dp_abs['P-values'] = df_dp_p_abs[brain_measure]
            df_dp_abs['T-values'] = df_dp_t_abs[brain_measure]

            print(df_int)
            print(df_ext)
            print(df_dp)
            print(df_int_abs)
            print(df_ext_abs)
            print(df_dp_abs)

            df_int.to_csv(brain_measure_p / 'results_int.csv')
            df_ext.to_csv(brain_measure_p / 'results_ext.csv')
            df_dp.to_csv(brain_measure_p / 'results_dp.csv')
            df_int_abs.to_csv(brain_measure_p / 'results_int_abs.csv')
            df_ext_abs.to_csv(brain_measure_p / 'results_ext_abs.csv')
            df_dp_abs.to_csv(brain_measure_p / 'results_dp_abs.csv')
            df_kernel.to_csv(brain_measure_p / 'best_kernels.csv')

            prediction_p = brain_measure_p / Path('predictions')
            if not prediction_p.is_dir():
                prediction_p.mkdir(
                        parents=True, exist_ok=True)
            for (region_name, region_df) in pred_dict_full[
                    model][brain_measure].items():
                region_name_p = prediction_p / Path(f'{region_name}')
                if not region_name_p.is_dir():
                    region_name_p.mkdir(parents=True, exist_ok=True)
                region_df.to_csv(region_name_p / 'predictions.csv')

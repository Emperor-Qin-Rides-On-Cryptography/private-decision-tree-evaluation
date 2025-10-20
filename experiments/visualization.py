# import os
#
# import matplotlib.pyplot as plt
# import pandas as pd
# import yaml
# from commons import *
#
# num_attributes={
#     "breast": 30,
#     "steel": 33,
#     "heart": 13,
#     "spam": 57,
# }
#
# def get_points_list(x, y):
#     assert len(x.values) == len(y.values)
#     return [(x.values[i], y.values[i]) for i in range(len(x.values))]
#
# def read_directory(dir_path):
#     res=[]
#     for path in os.listdir(dir_path):
#         # check if current path is a file
#         if os.path.isfile(os.path.join(dir_path, path)):
#             x = pd.read_csv(os.path.join(dir_path, path), names=['metric', 'value'], index_col=0).T
#             res.append(x)
#     if len(res) > 0:
#         return pd.concat(res)
#     else:
#         return None
#
# def read_sortinghats(dir_path):
#     res=[]
#     for path in os.listdir(dir_path):
#         # check if current path is a file
#         if os.path.isfile(os.path.join(dir_path, path)):
#             x = pd.read_csv(os.path.join(dir_path, path))
#             res.append(x)
#     if len(res) > 0:
#         return pd.concat(res)
#     else:
#         return None
#
# with open(os.path.join(PROJECT_ROOT,'experiments/experiment.yaml')) as f:
#     data = yaml.load(f, Loader=yaml.FullLoader)
#
#     dataset_name_list=data['plotting']['dataset']
#     results_path=f'experiments/results-{version_tag}'
#
#     # Create a dictionary with a list of upper and lower bounds for each dataset
#     time_bounds = {
#         'breast': [-10, 700],
#         'steel': [-10, 250],
#         'heart': None,
#         'spam': None,
#     }
#
#     comm_bounds = {
#         'breast': [800, 20000],
#         'steel': [900, 20000],
#         'heart': [300, 20000],
#         'spam': [1000, 26000],
#     }
#
#
#     for dataset_name in dataset_name_list:
#         print(dataset_name)
#
#         f1, plt_time=plt.subplots()
#         f2, plt_comm=plt.subplots()
#
#         # Get the current figure size in inches
#         current_fig_size = plt.gcf().get_size_inches()
#
#         # Set the desired aspect ratio: height = 0.5 * width
#         desired_width = current_fig_size[0]
#         desired_height = desired_width / 1.618
#
#         # Update the figure size with the new dimensions
#         f1.set_size_inches(desired_width, desired_height)
#         f2.set_size_inches(desired_width, desired_height)
#
#         # src1
#
#         raw_data=read_directory(os.path.join(PROJECT_ROOT, results_path, 'src1', dataset_name))
#
#         raw_data=raw_data[(2 <= raw_data["bitlength"]) & (raw_data["bitlength"]<=16)]
#
#         if raw_data is not None:
#             # Keep correct results, average them
#             raw_data=raw_data[raw_data["correctness"]==1]
#             data_avg=raw_data.groupby(['bitlength', 'hamming_weight', 'comparison'], as_index=False).mean()
#             data_std=raw_data.groupby(['bitlength', 'hamming_weight', 'comparison'], as_index=False).std()
#
#             # Folklore
#             folklore_data_avg=data_avg[data_avg["comparison"]==1]
#             folklore_data_std=data_std[data_std["comparison"]==1]
#
#             plt_time.fill_between(
#                 folklore_data_avg['bitlength'],
#                 folklore_data_avg['time_server_crypto'] - folklore_data_std['time_server_crypto'],
#                 folklore_data_avg['time_server_crypto'] + folklore_data_std['time_server_crypto'],
#                 alpha=0.2
#             )
#             plt_time.plot(folklore_data_avg['bitlength'], folklore_data_avg['time_server_crypto'], label=f'Folklore-PDTE')
#             plt_comm.plot(folklore_data_avg['bitlength'], folklore_data_avg['comm_query']/1000, label=f'Folklore-PDTE')
#
#             # Range cover
#             for hamming_weight in [2,4]:
#                 filtered_avg=data_avg[(data_avg["hamming_weight"]==hamming_weight) & (data_avg["comparison"]==0)]
#                 filtered_std=data_std[(data_std["hamming_weight"]==hamming_weight) & (data_std["comparison"]==0)]
#
#                 plt_time.fill_between(
#                     filtered_avg['bitlength'],
#                     filtered_avg['time_server_crypto'] - filtered_std['time_server_crypto'],
#                     filtered_avg['time_server_crypto'] + filtered_std['time_server_crypto'],
#                     alpha=0.2
#                 )
#                 plt_time.plot(filtered_avg['bitlength'], filtered_avg['time_server_crypto'], label=f'RCC-PDTE (h={hamming_weight})')
#                 plt_comm.plot(filtered_avg['bitlength'], filtered_avg['comm_query']/1000, label=f'RCC-PDTE (h={hamming_weight})')
#
#
#         #src2
#
#         raw_data_xcmp=read_directory(os.path.join(PROJECT_ROOT, results_path, 'src2', dataset_name))
#         raw_data_xcmp=raw_data_xcmp[(2 <= raw_data_xcmp["bitlength"] )&( raw_data_xcmp["bitlength"]<=16)]
#
#
#         if raw_data_xcmp is not None:
#
#             # Only valid entries
#             raw_data_xcmp = raw_data_xcmp[raw_data_xcmp['correctness']==1]
#             data_xcmp_avg=raw_data_xcmp.groupby(['bitlength', 'mult_path'], as_index=False).mean()
#             data_xcmp_std=raw_data_xcmp.groupby(['bitlength', 'mult_path'], as_index=False).std()
#
#             # Sum Path
#             filtered=data_xcmp_avg[data_xcmp_avg['mult_path']==0]
#             filtered_std=data_xcmp_std[data_xcmp_std['mult_path']==0]
#             plt_time.fill_between(
#                 filtered['bitlength'],
#                 filtered['time_server'] - filtered_std['time_server'],
#                 filtered['time_server'] + filtered_std['time_server'],
#                 alpha=0.2
#             )
#             plt_time.plot(filtered['bitlength'], filtered['time_server'], label=f'XXCMP-PDTE')
#             plt_comm.plot(filtered['bitlength'], filtered['comm_request']/1000, label=f'XXCMP-PDTE')
#
#             # Mult Path
#             # filtered=data_xcmp_avg[data_xcmp_avg['mult_path']==1]
#             # plt_time.fill_between(
#             #     filtered['bitlength'],
#             #     filtered['time_server'] - filtered_std['time_server'],
#             #     filtered['time_server'] + filtered_std['time_server'],
#             #     alpha=0.2
#             # )
#             # plt_time.plot(filtered['bitlength'], filtered['time_server'], label=f'XCMP (mult)')
#             # plt_comm.plot(filtered['bitlength'], filtered['comm_request']/1000, label=f'XCMP (mult)')
#
#         # SortingHats
#
#         #raw_data_sortinghats=read_sortinghats(os.path.join(PROJECT_ROOT, results_path, 'sortinghats', dataset_name))
#         raw_data_sortinghats=None
#         if raw_data_sortinghats is not None:
#             # Add approximated communication
#             raw_data_sortinghats['comm'] = raw_data_sortinghats['dataset'].apply(lambda x: num_attributes[x])*64*2048*2/8000
#             data_sortinghats_avg=raw_data_sortinghats.groupby(['dataset'], as_index=False).mean()
#             data_sortinghats_std=raw_data_sortinghats.groupby(['dataset'], as_index=False).std()
#
#             # Rewrite the line below, but make the dot black
#             plt_time.plot(11, data_sortinghats_avg['duration'], 'o', color='black', label=f'SortingHats')
#             plt_comm.plot(11, data_sortinghats_avg['comm'], 'o', color='black', label=f'SortingHats (Approx.)')
#
#
#         figures_dir=os.path.join(PROJECT_ROOT, results_path, 'figures')
#         if not os.path.exists(figures_dir):
#             os.makedirs(figures_dir)
#
#         plt_time.set_ylabel("毫秒")
#         plt_time.set_xlabel("比特精度")
#         plt_time.legend()
#         if time_bounds[dataset_name] is not None:
#             # set the yaxis range
#             plt_time.set_ylim(time_bounds[dataset_name][0], time_bounds[dataset_name][1])
#         plt_time.set_title(f"Inference Time for {dataset_name.capitalize()} dataset ({num_attributes[dataset_name]} attributes)")
#         f1.savefig(os.path.join(figures_dir, f"time-{dataset_name}.pdf"), bbox_inches = 'tight')
#         plt.close()
#
#         plt_comm.set_ylabel("千字节")
#         plt_comm.set_xlabel("比特精度")
#         plt_comm.legend()
#         plt_comm.set_yscale('log')
#         if comm_bounds[dataset_name] is not None:
#             # set the yaxis range
#             plt_comm.set_ylim(comm_bounds[dataset_name][0], comm_bounds[dataset_name][1])
#         plt_comm.set_title(f"Communication for {dataset_name.capitalize()} dataset ({num_attributes[dataset_name]} attributes)")
#         f2.savefig(os.path.join(figures_dir, f"comm-{dataset_name}.pdf"), bbox_inches = 'tight')
#         plt.close()
#
#         # # Comm-Comp Graphs
#         # for bitlength in [8, 11, 12, 16, 20 , 24, 36]:
#         #     f3, plt_time_comm = plt.subplots()
#         #     points = []
#
#         #     selected_data = data_avg[(data_avg['bitlength']==bitlength) & (data_avg['comparison']==1)]
#         #     plt_time_comm.plot(selected_data['comm_query'], selected_data['time_server_crypto'], 'o', label="Folklore")
#         #     points += get_points_list(selected_data['comm_query'], selected_data['time_server_crypto'])
#
#         #     selected_data = data_avg[(data_avg['bitlength']==bitlength) & (data_avg['comparison']==0)]
#         #     for hamming_weight in [4,6,8,16]:
#         #         filtered=selected_data[selected_data["hamming_weight"]==hamming_weight]
#         #         plt_time_comm.plot(filtered['comm_query'], filtered['time_server_crypto'], 'o', label=f"k={hamming_weight}")
#         #         points += get_points_list(filtered['comm_query'], filtered['time_server_crypto'])
#
#         #     selected_data_xcmp = data_xcmp_avg[(data_xcmp_avg['bitlength']==bitlength) & (data_xcmp_avg['mult_path']==0)]
#         #     plt_time_comm.plot(selected_data_xcmp['comm_request'], selected_data_xcmp['time_server'], 'o', label="XCMP")
#         #     points += get_points_list(selected_data_xcmp['comm_request'], selected_data_xcmp['time_server'])
#
#         #     if bitlength <= 11:
#         #         plt_time_comm.plot(data_sortinghats_avg['comm'], data_sortinghats_avg['duration'], 'o', label='SortingHats')
#         #         points += get_points_list(data_sortinghats_avg['comm'], data_sortinghats_avg['duration'])
#
#         #     # Throughput lines
#         #     # # MB/s -> KB/ms
#         #     # for throughput in [100, 1000, 10000, 100000]:
#         #     #     pareto_point=min(points, key=lambda x: x[0] + x[1]*throughput)
#         #     #     plt_time_comm.axline(pareto_point, slope=-1/throughput, linestyle='--', label=f'Throughput={throughput}')
#
#         #     plt_time_comm.legend()
#         #     plt_time_comm.set_title(f"bitlength={bitlength}, Throughputs in MB/s")
#         #     # plt_time_comm.set_xscale('log')
#         #     plt_time_comm.set_ylabel("Milliseconds")
#         #     plt_time_comm.set_xlabel("KBytes")
#         #     f3.savefig(os.path.join(figures_dir, f"time_comm/{dataset_name}-n={bitlength}.pdf"))
#
#         #     plt.close()
#
#

import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from commons import *

# --- 新增代码：设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# --- 修改结束 ---

# --- 新增代码：自定义颜色循环与注释参数 ---
import numpy as np

# 自定义颜色循环（要求的颜色）
custom_colors = ['#2F4F4F', '#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7']

# 注释参数：每条线上最多显示的数据点数量和点大小（可根据需要调整）
max_annotations_per_line = 10
marker_size = 6
# --- 修改结束 ---

# 自动缩放配置
use_percentile_clip = True
lower_percentile = 1.0
upper_percentile = 99.0
ylim_margin = 1.2


def auto_scale_ylim(ax, values, is_log=False, use_percentile=use_percentile_clip,
                    lowp=lower_percentile, highp=upper_percentile, margin=ylim_margin):
    """根据 values 自动设置 ax 的 y 轴范围。对数轴时会去除非正值。

    参数:
        ax: matplotlib 轴
        values: 可迭代的数值序列
        is_log: 轴是否为对数尺度
    """
    import math
    arr = np.array(values, dtype=float) if len(values) > 0 else np.array([])
    if arr.size == 0:
        return
    if is_log:
        arr = arr[arr > 0]
    if arr.size == 0:
        return
    if use_percentile:
        low = np.percentile(arr, lowp)
        high = np.percentile(arr, highp)
    else:
        low = arr.min()
        high = arr.max()
    if is_log:
        ymin = max(low / margin, arr[arr > 0].min() * 0.5, 1e-12)
    else:
        ymin = low / margin
    ymax = high * margin
    # 防止 ymin >= ymax
    if not math.isfinite(ymin) or not math.isfinite(ymax) or ymin >= ymax:
        return
    ax.set_ylim(ymin, ymax)

num_attributes = {
    "breast": 30,
    "steel": 33,
    "heart": 13,
    "spam": 57,
}

# --- 新增代码：数据集名称中英文映射 ---
dataset_chinese_names = {
    "breast": "乳腺癌",
    "steel": "钢材",
    "heart": "心脏病",
    "spam": "垃圾邮件",
}


# --- 修改结束 ---


def get_points_list(x, y):
    assert len(x.values) == len(y.values)
    return [(x.values[i], y.values[i]) for i in range(len(x.values))]


def annotate_points(ax, x_values, y_values, color=None, max_points=max_annotations_per_line, size=marker_size):
    """在给定的轴上为折线图注释最多 max_points 个点。

    参数:
        ax: matplotlib 的轴对象
        x_values, y_values: 可迭代的数据序列（pandas Series 或 numpy array）
        color: 注释和点的颜色
        max_points: 每条线上最多注释的点数
        size: 点的大小
    """
    # 转为 numpy array 便于索引
    x_arr = np.array(x_values)
    y_arr = np.array(y_values)
    n = len(x_arr)
    if n == 0:
        return

    # 选择要注释的索引：均匀间隔选取不超过 max_points
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points, dtype=int)

    for i in idx:
        ax.plot(x_arr[i], y_arr[i], 'o', color=color, markersize=size)
        # 在点旁边写具体数值，保留适当小数
        label = f"{y_arr[i]:.3f}" if isinstance(y_arr[i], (float, np.floating)) else f"{y_arr[i]}"
        # 偏移文本以免遮挡点
        ax.text(x_arr[i], y_arr[i], label, fontsize=8, color=color, verticalalignment='bottom', horizontalalignment='left')


def read_directory(dir_path):
    res = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            x = pd.read_csv(os.path.join(dir_path, path), names=['metric', 'value'], index_col=0).T
            res.append(x)
    if len(res) > 0:
        return pd.concat(res)
    else:
        return None


def read_sortinghats(dir_path):
    res = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            x = pd.read_csv(os.path.join(dir_path, path))
            res.append(x)
    if len(res) > 0:
        return pd.concat(res)
    else:
        return None


with open(os.path.join(PROJECT_ROOT, 'experiments/experiment.yaml')) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name_list = data['plotting']['dataset']
    results_path = f'experiments/results-{version_tag}'

    # Create a dictionary with a list of upper and lower bounds for each dataset
    time_bounds = {
        'breast': None,
        'steel': None,
        'heart': None,
        'spam': None,
    }

    comm_bounds = {
        'breast': None,
        'steel': None,
        'heart': None,
        'spam': None,
    }

    for dataset_name in dataset_name_list:
        print(dataset_name)

        f1, plt_time = plt.subplots()
        f2, plt_comm = plt.subplots()

        # Get the current figure size in inches
        current_fig_size = plt.gcf().get_size_inches()

        # Set the desired aspect ratio: height = 0.5 * width
        desired_width = current_fig_size[0]
        desired_height = desired_width / 1.618

        # Update the figure size with the new dimensions
        f1.set_size_inches(desired_width, desired_height)
        f2.set_size_inches(desired_width, desired_height)

        # src1
        raw_data = read_directory(os.path.join(PROJECT_ROOT, results_path, 'src1', dataset_name))

        raw_data = raw_data[(2 <= raw_data["bitlength"]) & (raw_data["bitlength"] <= 16)]

        # 用于收集 y 值以便自动缩放
        time_y_vals = []
        comm_y_vals = []

        if raw_data is not None:
            # Keep correct results, average them
            raw_data = raw_data[raw_data["correctness"] == 1]
            data_avg = raw_data.groupby(['bitlength', 'hamming_weight', 'comparison'], as_index=False).mean()
            data_std = raw_data.groupby(['bitlength', 'hamming_weight', 'comparison'], as_index=False).std()

            # prepare color iterator
            color_iter = iter(custom_colors)

            # Folklore
            folklore_data_avg = data_avg[data_avg["comparison"] == 1]
            folklore_data_std = data_std[data_std["comparison"] == 1]

            color = next(color_iter)
            plt_time.fill_between(
                folklore_data_avg['bitlength'],
                folklore_data_avg['time_server_crypto'] - folklore_data_std['time_server_crypto'],
                folklore_data_avg['time_server_crypto'] + folklore_data_std['time_server_crypto'],
                alpha=0.2, color=color
            )
            plt_time.plot(folklore_data_avg['bitlength'], folklore_data_avg['time_server_crypto'],
                          label=f'Folklore-PDTE', color=color)
            annotate_points(plt_time, folklore_data_avg['bitlength'], folklore_data_avg['time_server_crypto'], color=color)
            try:
                time_y_vals += list(folklore_data_avg['time_server_crypto'].dropna().astype(float).values)
            except Exception:
                pass
            plt_comm.plot(folklore_data_avg['bitlength'], folklore_data_avg['comm_query'] / 1000,
                          label=f'Folklore-PDTE', color=color)
            annotate_points(plt_comm, folklore_data_avg['bitlength'], folklore_data_avg['comm_query'] / 1000, color=color)
            try:
                comm_y_vals += list((folklore_data_avg['comm_query'] / 1000).dropna().astype(float).values)
            except Exception:
                pass

            # Range cover
            for hamming_weight in [2, 4]:
                filtered_avg = data_avg[(data_avg["hamming_weight"] == hamming_weight) & (data_avg["comparison"] == 0)]
                filtered_std = data_std[(data_std["hamming_weight"] == hamming_weight) & (data_std["comparison"] == 0)]

                color = next(color_iter, custom_colors[0])
                plt_time.fill_between(
                    filtered_avg['bitlength'],
                    filtered_avg['time_server_crypto'] - filtered_std['time_server_crypto'],
                    filtered_avg['time_server_crypto'] + filtered_std['time_server_crypto'],
                    alpha=0.2, color=color
                )
                plt_time.plot(filtered_avg['bitlength'], filtered_avg['time_server_crypto'],
                              label=f'FDC-PDTE (h={hamming_weight})', color=color)
                annotate_points(plt_time, filtered_avg['bitlength'], filtered_avg['time_server_crypto'], color=color)
                try:
                    time_y_vals += list(filtered_avg['time_server_crypto'].dropna().astype(float).values)
                except Exception:
                    pass
                plt_comm.plot(filtered_avg['bitlength'], filtered_avg['comm_query'] / 1000,
                              label=f'FDC-PDTE (h={hamming_weight})', color=color)
                annotate_points(plt_comm, filtered_avg['bitlength'], filtered_avg['comm_query'] / 1000, color=color)
                try:
                    comm_y_vals += list((filtered_avg['comm_query'] / 1000).dropna().astype(float).values)
                except Exception:
                    pass

        # (已移除 src2 / XXCMP 与 SortingHats 的绘图部分)

        figures_dir = os.path.join(PROJECT_ROOT, results_path, 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        # --- 修改图表1的标签为中文 ---
        plt_time.set_ylabel("毫秒")
        plt_time.set_xlabel("比特精度")
        plt_time.legend()
        #plt_time.set_yscale('log')
        # 自动缩放 y 轴（优先于静态 time_bounds）
        # try:
        #     auto_scale_ylim(plt_time, time_y_vals, is_log=True)
        # except Exception:
        #     pass
        if time_bounds[dataset_name] is not None:
            plt_time.set_ylim(time_bounds[dataset_name][0], time_bounds[dataset_name][1])
        # 使用中文名称设置标题
        plt_time.set_title(
            f"{dataset_chinese_names[dataset_name]}数据集上的推断时间 ({num_attributes[dataset_name]}个属性)")
        f1.savefig(os.path.join(figures_dir, f"time-{dataset_name}.pdf"), bbox_inches='tight')
        plt.close()

        # --- 修改图表2的标签为中文 ---
        plt_comm.set_ylabel("千字节 (KB)")
        plt_comm.set_xlabel("比特精度")
        plt_comm.legend()
        plt_comm.set_yscale('log')
        # 自动缩放 comm 图 y 轴
        try:
            auto_scale_ylim(plt_comm, comm_y_vals, is_log=True)
        except Exception:
            pass
        if comm_bounds[dataset_name] is not None:
            plt_comm.set_ylim(comm_bounds[dataset_name][0], comm_bounds[dataset_name][1])
        # 使用中文名称设置标题
        plt_comm.set_title(
            f"{dataset_chinese_names[dataset_name]}数据集上的通信开销 ({num_attributes[dataset_name]}个属性)")
        f2.savefig(os.path.join(figures_dir, f"comm-{dataset_name}.pdf"), bbox_inches='tight')
        plt.close()
        # --- 修改结束 ---

        # # Comm-Comp Graphs
        # for bitlength in [8, 11, 12, 16, 20 , 24, 36]:
        #     f3, plt_time_comm = plt.subplots()
        #     points = []

        #     selected_data = data_avg[(data_avg['bitlength']==bitlength) & (data_avg['comparison']==1)]
        #     plt_time_comm.plot(selected_data['comm_query'], selected_data['time_server_crypto'], 'o', label="Folklore")
        #     points += get_points_list(selected_data['comm_query'], selected_data['time_server_crypto'])

        #     selected_data = data_avg[(data_avg['bitlength']==bitlength) & (data_avg['comparison']==0)]
        #     for hamming_weight in [4,6,8,16]:
        #         filtered=selected_data[selected_data["hamming_weight"]==hamming_weight]
        #         plt_time_comm.plot(filtered['comm_query'], filtered['time_server_crypto'], 'o', label=f"k={hamming_weight}")
        #         points += get_points_list(filtered['comm_query'], filtered['time_server_crypto'])

        #     selected_data_xcmp = data_xcmp_avg[(data_xcmp_avg['bitlength']==bitlength) & (data_xcmp_avg['mult_path']==0)]
        #     plt_time_comm.plot(selected_data_xcmp['comm_request'], selected_data_xcmp['time_server'], 'o', label="XCMP")
        #     points += get_points_list(selected_data_xcmp['comm_request'], selected_data_xcmp['time_server'])

        #     if bitlength <= 11:
        #         plt_time_comm.plot(data_sortinghats_avg['comm'], data_sortinghats_avg['duration'], 'o', label='SortingHats')
        #         points += get_points_list(data_sortinghats_avg['comm'], data_sortinghats_avg['duration'])

        #     # Throughput lines
        #     # # MB/s -> KB/ms
        #     # for throughput in [100, 1000, 10000, 100000]:
        #     #     pareto_point=min(points, key=lambda x: x[0] + x[1]*throughput)
        #     #     plt_time_comm.axline(pareto_point, slope=-1/throughput, linestyle='--', label=f'Throughput={throughput}')

        #     plt_time_comm.legend()
        #     plt_time_comm.set_title(f"bitlength={bitlength}, Throughputs in MB/s")
        #     # plt_time_comm.set_xscale('log')
        #     plt_time_comm.set_ylabel("Milliseconds")
        #     plt_time_comm.set_xlabel("KBytes")
        #     f3.savefig(os.path.join(figures_dir, f"time_comm/{dataset_name}-n={bitlength}.pdf"))

        #     plt.close()

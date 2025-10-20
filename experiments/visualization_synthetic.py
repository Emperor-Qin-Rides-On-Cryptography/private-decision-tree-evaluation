import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from commons import *
import matplotlib.lines as mlines

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_points_list(x, y):
    assert len(x.values) == len(y.values)
    return [(x.values[i], y.values[i]) for i in range(len(x.values))]

def annotate_points(ax, x, y, fmt="{:.0f}", xytext=(3,3)):
    """Annotate points on axis `ax` at coordinates x,y using format `fmt`.
    x and y are iterables (pandas Series or lists)."""
    for xi, yi in zip(x, y):
        try:
            ax.annotate(fmt.format(yi), xy=(xi, yi), xytext=xytext, textcoords='offset points', fontsize=8)
        except Exception:
            # fallback: str conversion
            ax.annotate(str(yi), xy=(xi, yi), xytext=xytext, textcoords='offset points', fontsize=8)

def read_directory(dir_path):
    res=[]
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
    res=[]
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            x = pd.read_csv(os.path.join(dir_path, path))
            res.append(x)
    if len(res) > 0:
        return pd.concat(res)
    else:
        return None   

with open(os.path.join(PROJECT_ROOT,'experiments/experiment.yaml')) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

    results_path=f'experiments/results-{version_tag}'

    # unique colors for each bitlength to avoid duplicates in legend
    bitlength_color_list = [
        (8,  '#e41a1c'),  # red
        (12, '#377eb8'),  # blue
        (16, '#4daf4a'),  # green
        (26, '#984ea3'),  # purple
        (32, '#ff7f00'),  # orange
    ]

    raw_data=read_directory(os.path.join(PROJECT_ROOT, results_path, f'src1/synthetic'))
    comm_div=10**3
    if raw_data is not None:
        # Keep correct results, average them
        raw_data=raw_data[raw_data["correctness"]==1]
        raw_data['comm_query'] = raw_data['comm_query'] / comm_div
        raw_data['comm_response'] = raw_data['comm_response'] / comm_div
        data_avg=raw_data.groupby(['bitlength', 'num_attr', 'num_internal_nodes'], as_index=False).mean()
        data_std=raw_data.groupby(['bitlength', 'num_attr', 'num_internal_nodes'], as_index=False).std()

    # raw_data_xcmp=read_directory(os.path.join(PROJECT_ROOT, results_path, f'src2/synthetic'))
    raw_data_xcmp = None
    if raw_data_xcmp is not None:
        # Keep correct results, average them
        raw_data_xcmp = raw_data_xcmp[raw_data_xcmp['correctness']==1]
        raw_data_xcmp['comm_request'] = raw_data_xcmp['comm_request'] / comm_div
        raw_data_xcmp['comm_response'] = raw_data_xcmp['comm_response'] / comm_div
        data_xcmp_avg=raw_data_xcmp.groupby(['bitlength', 'mult_path', 'num_attr', 'inner_nodes'], as_index=False).mean()
        data_xcmp_std=raw_data_xcmp.groupby(['bitlength', 'mult_path', 'num_attr', 'inner_nodes'], as_index=False).std()
    
    # SortingHats data removed

    f1, plt_time=plt.subplots()
    f2, plt_comm=plt.subplots()
    
    
    ##################################################################################################################################
    
    # create legend for FDC and per-bitlength markers/colors
    FDC_label = mlines.Line2D([], [], color='black', linestyle='-', label='FDC-PDTE')
    # dynamic handles so colors match plotted lines and are unique
    bit_handles = []
    for bl, col in bitlength_color_list:
        bit_handles.append(mlines.Line2D([], [], color=col, marker='s', linestyle='None', label=f'n={bl}'))

    plt_time.add_artist(plt_time.legend(handles=[FDC_label], loc='upper left'))
    plt_time.add_artist(plt_time.legend(handles=bit_handles, loc='upper right'))
    plt_comm.add_artist(plt_comm.legend(handles=[FDC_label], loc='lower right'))
    plt_comm.add_artist(plt_comm.legend(handles=bit_handles, bbox_to_anchor=(0.62, 0), loc='lower center'))

    depth=6
    print(r"\toprule")
    print(r"Precision (bits) & FDC Approx. Runtime & FDC Communication & SortingHats Approx. Runtime \\")
    print(r"\midrule")
    
    for bitlength, color in bitlength_color_list:
        print("{:2.0f} & ".format(bitlength), end="")

        selected=data_avg[(data_avg['bitlength']==bitlength) & (data_avg['num_internal_nodes']==2**depth-1)]
        plt_time.plot(selected['num_attr'], selected['time_server_crypto'], color, label=f"FDC n={bitlength}")
        annotate_points(plt_time, selected['num_attr'], selected['time_server_crypto'], fmt="{:.0f}")
        plt_comm.plot(selected['num_attr'], selected['comm_query'], color, label=f"FDC n={bitlength}")
        annotate_points(plt_comm, selected['num_attr'], selected['comm_query'], fmt="{:.0f}")
        
        print("{:0.0f} - {:0.0f}".format(
            selected['time_server_crypto'].mean()-selected['time_server_crypto'].std(),
            selected['time_server_crypto'].mean()+selected['time_server_crypto'].std()
        ), end=" & ")

        # Print FDC communication (comm_query) mean ± std in the table
        try:
            if not selected.empty:
                comm_mean = selected['comm_query'].mean()
                comm_std = selected['comm_query'].std()
                print("{:0.0f} - {:0.0f}".format(
                    comm_mean - comm_std,
                    comm_mean + comm_std
                ), end=" & ")
            else:
                print(" - ", end=" & ")
        except Exception:
            # If something goes wrong (e.g. column missing), output a placeholder
            print(" - ", end=" & ")

        # Only FDC data printed (SortingHats/XXCMP removed)
        print(" - ", end="")
        
        print(r"\\")
            
    print(r"\bottomrule")
    

    figures_dir=os.path.join(PROJECT_ROOT, results_path, 'figures/sythetic')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    plt_time.set_ylabel("毫秒")
    plt_time.set_xlabel("属性数量")
    # plt_time.legend()
    plt_time.set_title("推理时间与属性数量关系")
    f1.savefig(os.path.join(figures_dir, f"time-depth-{depth}.pdf"), bbox_inches = 'tight')
    plt.close()

    plt_comm.set_ylabel("千字节")
    plt_comm.set_xlabel("属性数量")
    # plt_comm.legend()
    plt_comm.set_title("通信量与属性数量关系")
    plt_comm.set_yscale('log')
    f2.savefig(os.path.join(figures_dir, f"comm-depth-{depth}.pdf"), bbox_inches = 'tight')
    plt.close()
    
    ################################################################################################################################## 

    # create legend for FDC only and per-bitlength markers/colors (match first legend)
    FDC_label = mlines.Line2D([], [], color='black', linestyle='-', label='FDC-PDTE')
    bit_handles = []
    for bl, col in bitlength_color_list:
        bit_handles.append(mlines.Line2D([], [], color=col, marker='s', linestyle='None', label=f'n={bl}'))

    f1, plt_time=plt.subplots()
    plt_time.add_artist(plt_time.legend(handles=[FDC_label], loc='upper left'))
    plt_time.add_artist(plt_time.legend(handles=bit_handles, bbox_to_anchor = (0, 0.7), loc='center left'))

    f2, plt_comm=plt.subplots()
    plt_comm.add_artist(plt_comm.legend(handles=[FDC_label], loc='upper left'))
    plt_comm.add_artist(plt_comm.legend(handles=bit_handles, loc='upper right'))


    num_attr = 32
    print()
    print(r"\toprule")
    print(r"Precision (bits) & FDC-PDTE & SortingHats \\")
    print(r"\midrule")
    for bitlength, color in bitlength_color_list:
        print("{:2.0f} & ".format(bitlength), end="")

        selected_data=raw_data[(raw_data['bitlength']==bitlength) & (raw_data['num_attr']==num_attr) & (raw_data['num_internal_nodes']<=2000)]

        data_avg=selected_data.groupby(['num_internal_nodes', 'bitlength', 'num_attr'], as_index=False).mean()
        data_std=selected_data.groupby(['num_internal_nodes', 'bitlength', 'num_attr'], as_index=False).std()

        plt_time.fill_between(
            data_avg['num_internal_nodes'],
            data_avg['time_server_crypto']-data_std['time_server_crypto'],
            data_avg['time_server_crypto']+data_std['time_server_crypto'],
            color=color,
            alpha=0.1
        )
        plt_time.plot(data_avg['num_internal_nodes'], data_avg['time_server_crypto'], color, label=f"FDC n={bitlength}")
        annotate_points(plt_time, data_avg['num_internal_nodes'], data_avg['time_server_crypto'], fmt="{:.0f}", xytext=(5,2))
        plt_comm.plot(data_avg['num_internal_nodes'], data_avg['comm_query'], color, label=f"FDC n={bitlength}")
        annotate_points(plt_comm, data_avg['num_internal_nodes'], data_avg['comm_query'], fmt="{:.0f}", xytext=(5,2))
        
        print("{:0.0f}".format(        
            data_avg['comm_query'].mean()
            # data_avg['comm_query'].mean()-data_avg['comm_query'].std(),
            # data_avg['comm_query'].mean()+data_avg['comm_query'].std()
        ), end=" & ")

        # SortingHats/XXCMP removed; only FDC columns printed
        print("-", end="")
        
        print(r"\\")
            

    figures_dir=os.path.join(PROJECT_ROOT, results_path, 'figures/sythetic')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    plt_time.set_ylabel("毫秒")
    plt_time.set_xlabel("决策节点数量")
    plt_time.set_xscale('log')
    plt_time.set_xticks([4,8,16,32,64,128,256,512,1024], [4,8,16,32,64,128,256,512,1024])
    plt_time.set_ylim(-50,10000)
    # plt_time.legend()
    plt_time.set_title("推理时间与决策节点数量关系")
    f1.savefig(os.path.join(figures_dir, f"time-num-attr-{num_attr}.pdf"), bbox_inches = 'tight')
    plt.close()

    plt_comm.set_ylabel("千字节")
    plt_comm.set_xlabel("决策节点数量")
    # plt_comm.legend()
    plt_comm.set_title("通信量与决策节点数量关系")
    plt_comm.set_yscale('log')
    f2.savefig(os.path.join(figures_dir, f"comm-num-attr-{num_attr}.pdf"), bbox_inches = 'tight')
    plt.close()

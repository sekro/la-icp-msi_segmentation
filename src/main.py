# -*- coding: utf-8 -*-
"""
LA-ICP-MSI data analysis script for publication
###
2019-2021

@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""

import logging
import pandas as pd
import numpy as np
from laicpms_data_handler import LAICPMSData
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def main(clip=True, flip=True):
    """
    Main execution function - call this in your script to get the data & figure objects

    :param clip: bool - clip data according to setup()
    :param flip: bool - flip data orientation according to setup()
    :return: pandas.DataFrame, list[LAICPMSData], list[matplotlib.figure]
    """
    # main execution function
    d, ms = setup(clip, flip)
    # get rid of the sample info
    d = d.drop(columns=['sample'])
    figs = make_la_figs(ms)
    make_figure(d)
    return d, ms, figs


def setup(clip=True, flip=True):
    """
    Project specific data import and setup function
    :param clip: bool - use clipping
    :param flip: bool - use flipping
    :return: data as pandas dataframe, List[LAICPMSData obj]
    """

    # calibration
    # Zn: y = 0.0395 kcps/(µg/g)* x + 1.308 kcps
    # use inverse calibration function to get conc from counts; transformation m = 1/m and b = -1 * b/m
    calibration_functions = {
        'Zn:64': lambda x: 1/0.0395 * x - 1.308/0.0395,
    }
    # data files
    filenames = ["../data/LA_Data_C1SA1.csv",
                 "../data/LA_Data_C2SA1.csv",
                 "../data/LA_Data_C3SA1.csv",
                 "../data/LA_Data_C4SA1.csv",
                 "../data/LA_Data_C1SB1.csv",
                 "../data/LA_Data_C2SB1.csv",
                 "../data/LA_Data_C3SB1.csv",
                 "../data/LA_Data_C4SB1.csv",
                 "../data/LA_Data_C1SC1.csv",
                 "../data/LA_Data_C2SC1.csv",
                 "../data/LA_Data_C3SC1.csv",
                 "../data/LA_Data_C4SC1.csv"]
    # short sample names
    smpl_names = ["A_1",
                  "A_2",
                  "A_3",
                  "A_4",
                  "B_1",
                  "B_2",
                  "B_3",
                  "B_4",
                  "C_1",
                  "C_2",
                  "C_3",
                  "C_4"]
    # list on how to flip the data to get matching orientations, h = horizontally, v = vertically
    if flip:
        flip_list = [
            'h',
            'v',
            'h',
            'h',
            'h',
            'h',
            'v',
            'v',
            'v',
            'h',
            'h',
            'h'
        ]
    else:
        flip_list = ['no' for i in range(0, len(filenames))]

    # clip data to windows of defined size
    # main reason is comparability & tissue folds
    if clip:
        #100 px x 150 px
        clip_list = [
            (70,170,30,180),
            (70,170,30,180),
            (50,150,30,180),
            (60,160,50,200),
            (30,130,30,180),
            (40,140,30,180),
            (40,140,30,180),
            (40,140,30,180),
            (60,160,20,170),
            (60,160,20,170),
            (60,160,20,170),
            (60,160,20,170),
        ]
    else:
        clip_list = [None for i in range(0, len(filenames))]
    ms_data = []
    data = []
    # here the data gets processed into LAICPMSData objects - one per file
    # data contains all Zn:64 data - masked/segmented based on P:31 content
    for smpl, filename, clip, flip in zip(smpl_names, filenames, clip_list, flip_list):
        curr_ms_data = LAICPMSData(filename=filename, clip_data_around_center=clip, flip=flip, pixel_dimensions=(15,15))
        # only assign directly if you know what you are doing!
        curr_ms_data._calibration_functions = calibration_functions
        ms_data.append(curr_ms_data)
        data.append(curr_ms_data.get_masked_data(element_list=['Zn:64'], discriminator='P:31', only_on_tissue=True))
        data[-1]['sample'] = [smpl for i in range(0, len(data[-1]))]
    return pd.concat(data, ignore_index=True), ms_data


def make_legend(clib, with_mean_median_marker=True):
    legend_elements = []
    for name, color in clib.items():
        legend_elements.append(Patch(facecolor=color, label=name))
    if with_mean_median_marker:
        legend_elements.append(Line2D([0],[0], color='black', label='median', lw=2))
        legend_elements.append(Line2D([0],[0], color='w', marker='^', label='mean', markeredgecolor='black', markerfacecolor='black'))
    return legend_elements


def make_figure(all_data):
    """
    Makes the "main" distribution quantification figure
    :param all_data: input data to be plotted - pd.DataFrame
    :return: None
    """

    fig_box, ax_box = plt.subplots()
    box_width = 0.7
    ax_box, ax_dict = all_data.boxplot(return_type='both', sym='', notch=False, showmeans=True, patch_artist=True,
                                       widths=box_width)
    ax_box.set_ylabel('concentration [µg/g]')
    ax_box.yaxis.grid(False)
    ax_box.xaxis.grid(False)
    ymin, ymax = ax_box.get_ylim()
    xmin, xmax = ax_box.get_xlim()
    boxp_labels_pos = [2.5, 6.5, 10.5, 14.5]
    boxp_labels = ['Condition 1 - FF', 'Condition 2 - RTV', 'Condition 3 - FFix', 'Condition 4 - FFPS']
    for i in range(0, len(ax_dict['boxes']), len(boxp_labels)):
        ax_box.axhline(y=ymin, xmin=(i + 1 - box_width / 2 - xmin) / (xmax - xmin),
                       xmax=(i + len(boxp_labels) + box_width / 2 - xmin) / (xmax - xmin), color='black')

    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    ax_box.spines['bottom'].set_visible(False)

    ax_box.set_xticks(boxp_labels_pos, minor=False)
    ax_box.set_xticklabels(boxp_labels, minor=False)
    colorlib = {
        'total Zn': 'lightsteelblue',
        'Zn in pixels with high P (mostly gland epithelium)': 'moccasin',
        'Zn in pixels with low P (mostly stroma)': 'darkseagreen',
        'Zn in pixels with high Zn in lumen (most likely "stones")': 'thistle'
    }
    ax_box.legend(handles=make_legend(colorlib), loc='upper right')
    box_colors = ['lightsteelblue', 'moccasin', 'darkseagreen', 'thistle']
    repeat = int(len(ax_dict['boxes']) / len(box_colors))
    box_colors_long = []
    for i in range(0, repeat):
        box_colors_long = box_colors_long + box_colors
    for box, color in zip(ax_dict['boxes'], box_colors_long):
        box.set_facecolor(color)
        box.set_edgecolor('black')
    for whisker in ax_dict['whiskers']:
        whisker.set_color('black')
    for median, mean in zip(ax_dict['medians'], ax_dict['means']):
        median.set_color('black')
        median.set_linewidth = 2
        mean.set_markeredgecolor('black')
        mean.set_markerfacecolor('black')


def make_la_figs(ms_data, elements=['P:31', 'Zn:64'], no_batches=3, clip_equal_to_first=True, clip=(0, 0.95)):
    """
    Generates all LA-ICP-MS images & segmentation images
    :param ms_data: list of LAICPMSData objs
    :param elements: list of element strings to be plotted
    :param no_batches: number of "sets" / replicates
    :param clip_equal_to_first: Clip elemental data to optimal range of first data/element
    :param clip: tuple with upper and lower quantile to clip extreme values prior plotting
    :return: list of matplotlib figures
    """
    n = len(ms_data)
    if n % no_batches != 0:
        raise RuntimeError('number of ms data-sets / no_batches != 0')
    batch_size = int(n / no_batches)
    r = len(elements)
    figs = []
    for i in range(0, no_batches):
        figs.append(plt.subplots(r+1, batch_size))
    current_batch = 0
    abs_clip = {}
    for i, m in enumerate(ms_data):
        fig, ax = figs[current_batch]
        fig.suptitle("Elemental Distribution and Segmentation of {}".format(m._set))
        col = i % batch_size
        for row, e in enumerate(elements):
            data = m.get(e)
            if clip_equal_to_first:
                if e not in abs_clip:
                    abs_clip[e] = (np.quantile(data, clip[0]), np.quantile(data, clip[1]))
                pcm = ax[row, col].imshow(m._clip_data(data, clip_at=('abs', abs_clip[e][0], abs_clip[e][1])))
                if col == batch_size - 1:
                    fig.colorbar(pcm, ax=ax[row, col], extend='max', label='{}'.format(e))
                    ax[row, col].legend(
                        handles=[Patch(facecolor='red', label='Values above {}'.format(round(abs_clip[e][1], 2))),
                                 Patch(facecolor='blue', label='Values below {}'.format(round(abs_clip[e][0], 2)))],
                        bbox_to_anchor=(1.2, 1), loc='upper left',
                        borderaxespad=0.)
                clip_mask = np.zeros((data.shape[0], data.shape[1], 4), np.uint8)
                # clipped pixels > upper thresh = red ; < lower thresh = blue
                clip_mask[data > abs_clip[e][1]] = np.array([255,0,0,255])
                clip_mask[data < abs_clip[e][0]] = np.array([0,0,255,255])
            else:
                pcm = ax[row, col].imshow(m._clip_data(data, 'default'))
                clip_mask = np.zeros((data.shape[0], data.shape[1], 4), np.uint8)
                # clipped pixels > upper thresh = red ; < lower thresh = blue
                clip_mask[data > np.quantile(data, 0.75) + 4 * m._iqr(data)] = np.array([255, 0, 0, 255])
                fig.colorbar(pcm, ax=ax[row, col], extend='max')
            ax[row, col].imshow(clip_mask)
            if row == 0:
                ax[row, col].set_title('{}'.format(m.name))
            ax[row, col].xaxis.set_visible(False)
            ax[row, col].yaxis.set_visible(False)

        masked_img, colorlib = m.get_masked_image(element_list=['P:31'], return_as_uint8=False)
        ax[r, col].imshow(masked_img)
        ax[r, col].set_title('Segmentation')
        ax[r, col].xaxis.set_visible(False)
        ax[r, col].yaxis.set_visible(False)
        if col+1 == batch_size:
            ax[r, col].legend(handles=make_legend(colorlib, with_mean_median_marker=False), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        if (i + 1) % batch_size == 0:
            current_batch += 1
    return figs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    return_data = main()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.08,
                        hspace=0.14)
    plt.show()
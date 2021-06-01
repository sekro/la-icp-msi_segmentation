# -*- coding: utf-8 -*-
"""
LA-ICP-MSI data handler
###
2019-2020

@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""

import logging
import csv
import numpy as np
from matplotlib import cm
from scipy import ndimage
import cv2
import pandas as pd

logger = logging.getLogger(__name__)

class LAICPMSData:
    def __init__(self, filename=None, labels=None, time_and_date=None, elements=None, data=None,
                 clip_data_around_center=None, flip='h', pixel_dimensions=(1, 1)):
        self.name = None
        self.labels = labels
        self.time_and_date = time_and_date
        self.elements = elements
        self.data = data
        self.flip = flip
        self.clip_data = None
        self.pixel_dimensions = pixel_dimensions
        self.pixel_real_world_unit = 'µm'
        # dict self._calibration_functions: 'element_name': lambda x: 1.5 * x + 10
        # ... no checks - your lambda should function on np.arrays!!!
        self._calibration_functions = {}
        self.iqrs = {}
        self.filename = filename
        self.df = pd.DataFrame()
        self.shape = (0, 0)
        self.time_col = {}
        self.default_masks = {}
        self.on_tissue_mask = None
        self.lumen_mask = None
        self.ignore_pixel = None
        self._img = None
        self._condition = None
        self._set = None
        self._sample = None
        self._map_default_threshold_params = {
            'Zn:64': {'factor': 1.5, 'percentage': 1},
            'P:31': {'factor': 0.1, 'percentage': 1}
        }
        if self.filename is not None:
            self._extract_laicpms_data_from_csv()
            self._calc_iqrs()
            self._build_dataframes()
            self._build_default_masks()
            self._auto_build_on_tissue_mask()
            self.ignore_pixel = np.zeros(self.shape).astype(np.bool)
        if clip_data_around_center is not None:
            if isinstance(clip_data_around_center, float):
                if 0 < clip_data_around_center <= 1:
                    h, w = self.shape
                    self.clip_data = (int(h / 2 - h * clip_data_around_center),
                                      int(h / 2 + h * clip_data_around_center),
                                      int(w / 2 - w * clip_data_around_center),
                                      int(w / 2 + w * clip_data_around_center))
            if isinstance(clip_data_around_center, tuple):
                if (len(clip_data_around_center) == 4) and all([isinstance(x, int) for x in clip_data_around_center]):
                    self.clip_data = clip_data_around_center
            if self.clip_data is not None:
                self.ignore_pixel.fill(True)
                self.ignore_pixel[self.clip_data[0]:self.clip_data[1], self.clip_data[2]:self.clip_data[3]] = False

    def real_world_x(self, tick_val, tick_pos):
        # matplotlib FuncFormatter compatible
        return tick_val * self.pixel_dimensions[0]

    def real_world_y(self, tick_val, tick_pos):
        # matplotlib FuncFormatter compatible
        return tick_val * self.pixel_dimensions[1]

    def get(self, element, get_clipped=False, get_raw=False):
        """
        gets the data for an element
        :param element: element string as in raw file
        :param get_clipped: all or clipped data
        :param get_raw: only relevant if calibration function present - if true return raw data
        :return: numpy array of the data or None
        """
        if element in self.elements:
            if get_clipped:
                return_data = np.flip(self.data[:, self.elements[element]].transpose(), 0)[
                              self.clip_data[0]:self.clip_data[1], self.clip_data[2]:self.clip_data[3]]
            else:
                if self.flip == 'h':
                    return_data = np.flip(self.data[:, self.elements[element]].transpose(), 0)
                elif self.flip == 'v':
                    return_data = np.flip(self.data[:, self.elements[element]].transpose(), 1)
                elif self.flip == 'vh' or self.flip == 'hv':
                    return_data = np.flip(np.flip(self.data[:, self.elements[element]].transpose(), 1), 0)
                else:
                    return_data = self.data[:, self.elements[element]].transpose()
            if element in self._calibration_functions and not get_raw:
                return self._calibration_functions[element](return_data)
            else:
                return return_data
        elif element in self.time_col:
            return self.data[:, self.time_col[element]]
        else:
            return None

    def _extract_laicpms_data_from_csv(self, std_label=True):
        """
        load data into lists - only implemented and tested for shimadzu csv
        """
        data = []
        with open(self.filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    self.labels = row
                if i == 1:
                    self.time_and_date = row
                if i == 2:
                    elements = row
                if i > 2:
                    data.append(row)
        data = np.array(data)
        data = data.astype(np.float)
        col_dict = {}
        for i, ele in enumerate(elements):
            if ele in col_dict:
                col_dict[ele].append(i)
            else:
                col_dict[ele] = [i]
        self.time_col[self.labels[0]] = col_dict.pop('')
        self.elements = col_dict
        self.data = data
        last_shape = None
        current_shape = None
        for element in self.elements.keys():
            current_shape = self.get(element).shape
            if last_shape is not None:
                if current_shape != last_shape:
                    logger.warning(
                        'Oops data has different shape - that should not happen! Better check shapes manually')
                    break
            last_shape = current_shape
        self.shape = current_shape
        # std_label = True assumes something like self.labels[1] = "Condition 1 Set A Sample 001_1"
        if std_label:
            splitted_label = str.split(self.labels[1], ' ')
            if len(splitted_label) == 6:
                self._condition = splitted_label[0] + ' ' + splitted_label[1]
                self._set = splitted_label[2] + ' ' + splitted_label[3]
                self._sample = splitted_label[4] + ' ' + splitted_label[5]
                self.name = self._condition
            else:
                self.name = self.labels[1]
        else:
            self.name = self.labels[1]
        logger.info('Data loaded from {}'.format(self.filename))

    def _build_dataframes(self):
        """
        Builds one data frame with flattened arrays and X and Y columns holding the coordinates
        :return:
        """
        y, x = self.shape
        self.df['X'] = np.tile([e for e in range(0, x)], y)
        self.df['Y'] = np.repeat([e for e in range(0, y)], x)
        for element in self.elements.keys():
            self.df[element] = self.get(element).flatten()

    def _calc_iqrs(self):
        for element in self.elements.keys():
            data = self.get(element)
            self.iqrs[element] = self._iqr(data)

    def _iqr(self, data):
        return np.quantile(data, 0.75) - np.quantile(data, 0.25)

    def _threshold_and_contour(self, gray, gblur_kernel=(5, 5),
                               with_morph=False, morph_kernel=(2, 2), morph_mode=cv2.MORPH_OPEN, morph_iterations=1):
        """
        Blurs & Otsu thresholds the 8-bit grayscale input image with optional morphing - followed by contour detection
        :param gray: 8-bit grayscale image
        :param gblur_kernel: tuple gaussian kernel - odd-numbered, symmertric or unsymmetric
        :param with_morph: bool
        :param morph_kernel: tuple morph kernel
        :param morph_mode: cv2.MORPH_OPEN or cv2.MORPH_CLOSE
        :param morph_iterations: integer
        :return: (contours, hierarchy)
        """
        gray = cv2.GaussianBlur(gray, gblur_kernel, 0)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if with_morph:
            kernel = np.ones(morph_kernel, np.uint8)
            thresh = cv2.morphologyEx(thresh, morph_mode, kernel, iterations=morph_iterations)
        return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def _build_default_masks(self, clip_at=(0.01, 0.99)):
        """
        Builds masks per element after loading data from csv - customize to you needs
        :return:
        """
        for element in self.elements.keys():
            if element == 'P:31':
                self.default_masks[element] = self.get_mask(element=element, clip_at=(0.05, 0.95), inverted=True,
                                                            use_cv2=True)
            else:
                self.default_masks[element] = ndimage.binary_dilation(ndimage.binary_opening(
                    ndimage.binary_dilation(self.get_mask(element=element, clip_at=clip_at, inverted=True)),
                    structure=np.ones((2, 2))))

    def _filter_contours_by_data(self, contours, data, threshold, around_threshold=None,
                                 kernel=np.ones((5, 5), np.uint8)):
        """
        Contour filtering based on data inside contour and optionally "around" -> dilated contour
        :param contours: list of contours to be filtered
        :param data: the data used for thresholding during filtering
        :param threshold: the quantile to threshold the data (currently hard coded to median(data) < threshold)
        :param around_threshold: the quantile to optionally threshold "around" -> around > threshold
        :param kernel: defines the dilation for "around" thresholding
        :return: filtered contours as list
        """
        filtered_contours = []
        for c in contours:
            tmp = np.zeros(self.shape).astype(np.uint8)
            cv2.drawContours(tmp, [c], -1, (255), -1)
            cv2.drawContours(tmp, [c], -1, (0), 1)
            tmp = tmp > 0
            if data[tmp].size != 0:
                if np.nanmedian(data[tmp]) < np.quantile(data, threshold):
                    if around_threshold is not None:
                        around = cv2.bitwise_xor(cv2.dilate(tmp.astype(np.uint8), kernel, iterations=1),
                                                 tmp.astype(np.uint8))
                        if np.nanmedian(data[around.astype(np.bool)]) > np.quantile(data, around_threshold):
                            filtered_contours.append(c)
                    else:
                        filtered_contours.append(c)
        return filtered_contours

    def _auto_build_on_tissue_mask(self, element='P:31', clip_at=(0.25, 0.5), min_area_factor=0.0005,
                                   gblur_kernel=(5, 5)):
        """
        On default settings this function uses the P:31 signal with a high contrast - data < 25 percentile = 0,
        data > 50 percentile == 1 followed by gaussian blurring Otsu thresholding and contour filtering to find the
        tissue and potential lumens
        :param element: check what you can use - P:31 works well for prostate
        :param clip_at: improve the contrast of the raw data -> thresholding works on 8bit image! data prop > 8 bit
        :param min_area_factor: percentage of image size that is used to calc threshold for min size of contours
        :param gblur_kernel: odd values - purpose noise removal prior Otsu thresholding
        :return:
        """
        if element in self.elements:
            self._img = self.norm_to_8bit_grey(element=element, clip_at=clip_at)
            contours, hierarchy = self._threshold_and_contour(self._img, gblur_kernel=gblur_kernel)
            child_contours = []
            min_area = min_area_factor * self.shape[0] * self.shape[1]
            areas = [cv2.contourArea(c) for c in contours]
            max_idx = np.argmax(areas)
            children_idx = self._get_all_children(hierarchy, max_idx)
            for i, cnt in enumerate(contours):
                if cv2.contourArea(cnt) >= min_area and i in children_idx:
                    child_contours.append(cnt)
            child_contours = self._filter_contours_by_data(child_contours, data=self.get(element), threshold=0.45,
                                                           around_threshold=0.55)
            self._img = cv2.cvtColor(self._img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(self._img, child_contours, -1, (0, 0, 255), 1)
            cv2.drawContours(self._img, contours, max_idx, (0, 255, 0), 1)
            tissue_mask = np.zeros(self.shape)
            lumen_mask = np.zeros(self.shape)
            cv2.drawContours(tissue_mask, [contours[max_idx]], -1, (255), -1)
            cv2.drawContours(tissue_mask, child_contours, -1, (0), -1)
            cv2.drawContours(lumen_mask, child_contours, -1, (255), -1)
            self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGBA)
            self.lumen_mask = lumen_mask > 0
            self.on_tissue_mask = tissue_mask > 0

    def _get_all_children(self, h, parent_id):
        """
        small helper function to manage contour hierarchies from OpenCV
        :param h: contour hierarchy
        :param parent_id: parent id
        :return: list of contour indexes
        """
        return list(np.where(h[:, :, 3].flatten() == parent_id)[0])

    def get_iqr(self, element):
        if element in self.elements:
            return self.iqrs[element]
        else:
            return None

    def get_mask(self, element, clip_at=None, factor=0.1, percentage=1, inverted=False, use_defaults=True,
                 use_cv2=False,
                 min_area_factor=0.0005, min_data_percentile=0.5):
        """

        :param element: data to build the mask for
        :param clip_at: pre-process the raw data prior mask calculation via clipping - provide a tuple with lower and
                        upper percentile - e.g. (0.01, 0.99)
        :param factor: modifies the threshold for zscore based masking:
                       threshold = percentage * (upper quartile + factor * inter quartile range)
        :param percentage: modifies the threshold for zscore based masking
        :param inverted: return inverted mask or not
        :param use_defaults: bool
        :param use_cv2: bool - if true uses cv2 Otsu thresholding + contours to get mask
        :param min_area_factor: min area % of total img / data size to threshold contours
        :param min_data_percentile: pixel has to be above this value to be masked
        :return: bool mask 2d with shape of data
        """
        if element in self.elements:
            data = self.get(element)
            if clip_at:
                data = self._clip_data(data, clip_at)
            if use_cv2:
                gray = self.norm_to_8bit_grey(element=element, clip_at=clip_at)
                contours, hierarchy = self._threshold_and_contour(gray, with_morph=True)
                filtered_contours = []
                min_area = min_area_factor * self.shape[0] * self.shape[1]
                for cnt in contours:
                    if cv2.contourArea(cnt) >= min_area:
                        filtered_contours.append(cnt)
                tmp_mask = np.zeros(self.shape)
                cv2.drawContours(tmp_mask, filtered_contours, -1, (255), -1)
                return np.logical_and(tmp_mask > 0, data > np.quantile(data, min_data_percentile))
            else:
                if use_defaults and (element in self._map_default_threshold_params):
                    percentage = self._map_default_threshold_params[element]['percentage']
                    factor = self._map_default_threshold_params[element]['factor']
                data_threshold = percentage * (np.quantile(data, 0.75) + factor * self._iqr(data))
                if inverted:
                    return data > data_threshold
                else:
                    return data < data_threshold
        else:
            return None

    def print_stats(self, elements=None):
        if elements is None:
            elements = [e for e in self.elements.keys()]
        for element in elements:
            if element in self.elements:
                self._print_element_stats(element)
            else:
                print('Element {} not in this dataset!'.format(element))

    def _print_element_stats(self, element):
        data = self.get(element)
        print('---stats for {}---'.format(element))
        print('min, max, mean, median, std')
        print('{},{},{},{},{}'.format(np.min(data), np.max(data), np.mean(data), np.median(data), np.std(data)))
        print('--------')

    def _clip_data(self, data, clip_at):
        a_min, a_max = (0, 1)
        if clip_at == 'default':
            a_max = np.quantile(data, 0.75) + 4 * self._iqr(data)
            a_min = np.min(data)
        elif isinstance(clip_at, float):
            a_max = np.quantile(data, clip_at)
            a_min = np.min(data)
        elif isinstance(clip_at, tuple):
            if len(clip_at) == 2:
                a_min = np.quantile(data, clip_at[0])
                a_max = np.quantile(data, clip_at[1])
            elif len(clip_at) == 3:
                if clip_at[0] == 'abs':
                    a_min = clip_at[1]
                    a_max = clip_at[2]
        else:
            a_min = np.min(data)
            a_max = clip_at
        return np.clip(data, a_min=a_min, a_max=a_max)

    def norm_to_8bit_grey(self, element, clip_at=None):
        if element in self.elements:
            data = self.get(element)
            if clip_at:
                data = self._clip_data(data, clip_at)
            data = data - np.min(data)
            return np.round(255 * data / np.max(data), 0).astype(np.uint8).reshape(data.shape[0], data.shape[1], 1)
        else:
            return None

    def get_masked_image(self, element_list, show_only_on_lumen=None, use_default_masks=True, only_in_tissue=True,
                         separate_color_for_lumen=False, cmap='viridis', return_as_uint8=True):
        if show_only_on_lumen is None:
            show_only_on_lumen = []
        extra_colors = 3
        if separate_color_for_lumen:
            step_size = 2
        else:
            step_size = 1
        no_element_idx = step_size * len(element_list)
        no_colors = no_element_idx + extra_colors
        tissue_index = no_element_idx + 1
        lumen_index = no_element_idx + 2
        off_tissue_index = 0
        fcmap = cm.get_cmap(cmap, no_colors)
        norm = cm.colors.Normalize(vmax=no_colors - 1, vmin=0)
        img = np.zeros(self.shape)
        element_to_rbga_map = {}
        if return_as_uint8:
            element_to_rbga_map['tissue'] = np.round((np.array(fcmap(norm(tissue_index))) * 255), 0).astype(np.uint8)
            element_to_rbga_map['lumen'] = np.round((np.array(fcmap(norm(lumen_index))) * 255), 0).astype(np.uint8)
            element_to_rbga_map['off_tissue/clipped'] = np.round((np.array(fcmap(norm(off_tissue_index))) * 255),
                                                                 0).astype(np.uint8)
        else:
            element_to_rbga_map['tissue'] = np.array(fcmap(norm(tissue_index)))
            element_to_rbga_map['lumen'] = np.array(fcmap(norm(lumen_index)))
            element_to_rbga_map['off_tissue/clipped'] = np.array(fcmap(norm(off_tissue_index)))
        for i in range(0, no_element_idx, step_size):
            if return_as_uint8:
                if separate_color_for_lumen:
                    element_to_rbga_map['high {}'.format(element_list[int(i / step_size)])] = np.round(
                        (np.array(fcmap(norm(i + 1))) * 255), 0).astype(np.uint8)
                    element_to_rbga_map['high {} (luminal)'.format(element_list[int(i / step_size)])] = np.round(
                        (np.array(fcmap(norm(i + 2))) * 255), 0).astype(np.uint8)
                else:
                    if element_list[int(i / step_size)] in show_only_on_lumen:
                        element_to_rbga_map['high {} (luminal)'.format(element_list[int(i / step_size)])] = np.round(
                            (np.array(fcmap(norm(i + 1))) * 255), 0).astype(np.uint8)
                    else:
                        element_to_rbga_map['high {}'.format(element_list[int(i / step_size)])] = np.round(
                            (np.array(fcmap(norm(i + 1))) * 255), 0).astype(np.uint8)
            else:
                if separate_color_for_lumen:
                    element_to_rbga_map['high {}'.format(element_list[int(i / step_size)])] = np.array(fcmap(norm(i + 1)))
                    element_to_rbga_map['high {} (luminal)'.format(element_list[int(i / step_size)])] = np.array(fcmap(norm(i + 2)))
                else:
                    if element_list[int(i / step_size)] in show_only_on_lumen:
                        element_to_rbga_map['high {} (luminal)'.format(element_list[int(i / step_size)])] = np.array(
                            fcmap(norm(i + 1)))
                    else:
                        element_to_rbga_map['high {}'.format(element_list[int(i / step_size)])] = np.array(fcmap(norm(i + 1)))
        if use_default_masks:
            if only_in_tissue:
                img[self.on_tissue_mask] = tissue_index
                img[self.lumen_mask] = lumen_index
                for i in range(0, no_element_idx, step_size):
                    if separate_color_for_lumen:
                        img[self.default_masks[element_list[int(i / step_size)]] & (
                                self.on_tissue_mask & ~self.lumen_mask)] = i + 1
                        img[self.default_masks[element_list[int(i / step_size)]] & self.lumen_mask] = i + 2
                    else:
                        if element_list[int(i / step_size)] in show_only_on_lumen:
                            img[self.default_masks[element_list[int(i / step_size)]] & self.lumen_mask] = i + 1
                        else:
                            img[self.default_masks[element_list[int(i / step_size)]] & (
                                    self.on_tissue_mask | self.lumen_mask)] = i + 1
            else:
                for i, element in enumerate(element_list):
                    img[self.default_masks[element]] = i + 1
            img[self.ignore_pixel] = 0
        if return_as_uint8:
            return np.round((fcmap(norm(img)) * 255), 0).astype(np.uint8), element_to_rbga_map
        else:
            return fcmap(norm(img)), element_to_rbga_map

    def get_masked_data(self, element_list, discriminator, use_default_masks=True, only_on_tissue=True):
        tmp_list = []
        if use_default_masks:
            if discriminator in self.elements:
                for element in element_list:
                    if element in self.elements:
                        data = self.get(element)
                        if only_on_tissue:
                            tmp_list.append(pd.DataFrame({self.name + '_' + element + 'total_on_tissue_and_lumen': data[
                                (self.on_tissue_mask | self.lumen_mask) & ~self.ignore_pixel]}))
                            tmp_list.append(pd.DataFrame({
                                self.name + '_' + element + '_high_' + discriminator + '_on_tissue':
                                    data[self.default_masks[
                                             discriminator] & self.on_tissue_mask & ~self.ignore_pixel]}))
                            tmp_list.append(pd.DataFrame({
                                self.name + '_' + element + '_low_' + discriminator + '_on_tissue':
                                    data[(~self.default_masks[
                                        discriminator]) & self.on_tissue_mask & ~self.ignore_pixel]}))
                            tmp_list.append(pd.DataFrame({self.name + '_' + 'high_' + element + '_in_lumen': data[
                                self.default_masks[element] & self.lumen_mask & ~self.ignore_pixel]}))
                        else:
                            tmp_list.append(pd.DataFrame({self.name + '_' + element + '_high_' + discriminator: data[
                                self.default_masks[discriminator] & ~self.ignore_pixel]}))
                            tmp_list.append(pd.DataFrame({self.name + '_' + element + '_low_' + discriminator: data[
                                ~self.default_masks[discriminator] & ~self.ignore_pixel]}))
        return pd.concat(tmp_list, axis=1)

    def info(self):
        for element in self.elements.keys():
            print('The element {} data has a shape of {}'.format(element, self.get(element).shape))
        for element in self.time_col.keys():
            print('The element {} data has a shape of {}'.format(element, self.get(element).shape))
        print('The data has a per element shape of {}'.format(self.shape))
        print('The dataframe stats looks like this:\n{}'.format(self.df.describe()))
        print('Default masks:\n{}'.format(self.default_masks))
        print('On tissue mask:\n{}'.format(self.on_tissue_mask))

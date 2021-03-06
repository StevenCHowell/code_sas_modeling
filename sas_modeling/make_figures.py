#!/usr/bin/env python
# coding:utf-8
"""
    Author:  Steven C. Howell --<steven.howell@nist.gov>
    Purpose: collection of helpful bokeh plotting functions
    Created: 12/22/2016

00000000011111111112222222222333333333344444444445555555555666666666677777777778
12345678901234567890123456789012345678901234567890123456789012345678901234567890
"""
from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import pandas as pd

import bokeh.plotting
import bokeh.io
# from bokeh.palettes import Colorblind8 as palette
import bokeh.layouts

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

try:
    import datashader
    from datashader.bokeh_ext import InteractiveImage
except ImportError:
    logging.warning('datashader failed to import, `ds_lines` not available')


def define_solarized(n=11):
    palette = [
        '#0087ff',  # blue
        '#d70000',  # red
        '#5f8700',  # green
        '#af8700',  # yellow
        '#af005f',  # magenta
        '#00afaf',  # cyan
        '#d75f00',  # orange
        '#5f5faf',  # violet
        '#262626',  # base02
        '#585858',  # base01
        '#8a8a8a',  # base1
        # '#626262',  # base00
        # '#808080',  # base0
        # '#e4e4e4',  # base2
        # '#ffffd7',  # base3
        # '#1c1c1c',  # base03
        ]

    if n > len(palette):
        n = len(palette)

    return palette[:n]


def errorbar(fig, x, y, xerr=None, yerr=None, color='red',
             point_kwargs={}, error_kwargs={}):

    fig.circle(x, y, color=color, **point_kwargs)

    if xerr is not None:
        x_err_x = []
        x_err_y = []
        for px, py, err in zip(x, y, xerr):
            x_err_x.append((px - err, px + err))
            x_err_y.append((py, py))
        fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

    if yerr is not None:
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(x, y, yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))
        fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)


def round_to_n(x, n=3):
    return round(x, -int(np.floor(np.log10(x))) + (n - 1))


def plot_guinier_fit(q2, log_iq, fit_line, i0, i0_err, rg, rg_err, dlog_iq,
                     q_range, xerr=None, save_fname='fit_comparison.html'):
    '''
    plot data and a fit line
    '''

    n_round = 6

    i0 = round_to_n(i0, n=n_round)
    rg = round_to_n(rg, n=n_round)
    if i0_err == 0:
        i0_err = 'NA'
    else:
        i0_err = round_to_n(i0_err, n=n_round)

    if rg_err == 0:
        rg_err = 'NA'
    else:
        rg_err = round_to_n(rg_err, n=n_round)

    title = '{}, I(0) = {}+/-{}, Rg = {}+/-{}, q-range: [{}, {}]'.format(
        save_fname.split('.')[0], i0, i0_err, rg, rg_err, q_range[0],
        q_range[1])
    p = bokeh.plotting.figure(title=title, x_axis_label='q^2 (1/A^2)',
                              y_axis_label='ln(I(q))', width=472, height=400)

    p.line(q2, fit_line, color=palette[0], legend="fit", line_width=2)

    if xerr is None:
        errorbar(p, q2[1:], log_iq[1:], yerr=dlog_iq[1:], color=palette[1],
                 point_kwargs={'legend': 'raw'})
        errorbar(p, q2[:1], log_iq[:1], yerr=dlog_iq[:1], color=palette[3],
                 point_kwargs={'legend': 'extrapolated'})
    else:
        NotImplemented

    bokeh.io.output_file(save_fname)
    bokeh.io.save(p)
    bokeh.io.reset_output()
    return p


def plot_iq_and_guinier(q, iq, diq, save_fname='I(q)_and_guinier.html'):
    '''
    plot data using linear, log, and Guinier axes
    '''

    p0 = bokeh.plotting.figure(title='linear', x_axis_label='q (1/A)',
                               y_axis_label='I(q)')
    errorbar(p0, q, iq, yerr=diq, color=palette[1])

    p1 = bokeh.plotting.figure(title='log', x_axis_label='q (1/A)',
                               y_axis_label='I(q)', x_axis_type='log',
                               y_axis_type='log',
                               )
    errorbar(p1, q, iq, yerr=diq, color=palette[1])

    x = q ** 2
    y = np.log(iq)

    p2 = bokeh.plotting.figure(title='Guinier', x_axis_label='q^2 (1/A^2)',
                               y_axis_label='ln(I(q))')

    dy_skew = (np.log(iq + diq) - np.log(iq - diq)) / 2.0

    errorbar(p2, x, y, yerr=dy_skew, color=palette[0],
             point_kwargs={'legend': 'log(iq +/- averaged diq)'})

    dy = diq / iq
    errorbar(p2, x, y, yerr=dy, color=palette[1],
             point_kwargs={'legend': 'log(iq +/- diq)'})

    r0 = [p0, p1]
    r1 = [p2]  # , p3]

    layout = bokeh.layouts.gridplot([r0, r1])

    bokeh.io.output_file(save_fname)
    bokeh.io.save(layout)
    bokeh.io.reset_output()
    return layout


def np_to_ds_format(x_array, y_arrays):
    """
    reformat data for plotting in DataShader

    Parameters
    ----------
    x_array: array of x-values (should be a one dimensional np.array of length
             N)
    y_arrays: M different arrays of y-values (should be a np.array with
              dimensions
        NxM) L)

    Returns
    -------
    ds_df: datashader DataFrame
    """

    n_points, n_arrays = y_arrays.shape
    assert n_points == len(x_array), 'mismatched x/y-data'

    dfs = []
    split = pd.DataFrame({'x': [np.nan]})
    for i in range(n_arrays):
        x = x_array
        y = y_arrays[:, i]
        df = pd.DataFrame({'x': x, 'y': y})
        dfs.append(df)
        dfs.append(split)

    ds_df = pd.concat(dfs, ignore_index=True)
    return ds_df


def ds_lines(ds_df, plot=None):
    """
    plot many lines using datashader

    Parameters
    ----------
    ds_df: see np_to_ds_format

    Returns
    -------
    img: datashader image object
    """

    x_range = ds_df['x'].min(), ds_df['x'].max()
    y_range = ds_df['y'].min(), ds_df['y'].max()

    if plot is not None:
        plot_width = plot.plot_width
        plot_height = plot.plot_height
    else:
        plot_width = 400
        plot_height = 300

    def plot_data(x_range=x_range, y_range=y_range, plot_height=plot_height,
                  plot_width=plot_width, aggregator=datashader.count()):
        canvas = datashader.Canvas(x_range=x_range, y_range=y_range,
                                   plot_height=plot_height,
                                   plot_width=plot_width)
        agg = canvas.line(ds_df, 'x', 'y', aggregator)
        img = datashader.transfer_functions.shade(agg, how='eq_hist')
        return img

    if plot is not None:
        plot.x_range = bokeh.models.Range1d(*x_range)
        plot.y_range = bokeh.models.Range1d(*y_range)
        InteractiveImage(plot, plot_data, aggregator=datashader.count())
    else:
        return plot_data(x_range, y_range)


solarized = define_solarized()
palette = solarized

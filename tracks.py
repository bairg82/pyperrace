# This stores trafck definitions

import numpy as np


def set_track_params(track_name):
    # Setting paramters for given track name
    # adding new track:
    # always add startline, endline, sections, track_file, trk_col for a track
    if track_name == 'h1':
        # as hungaroring turn 1
        startline = np.array([200, 220, 200, 50])
        endline = np.array([200, 1250, 250, 1400])
        trk_col = np.array([99, 99, 99])  # pálya színe (szürke)
        track_inside_color = np.array([255, 0, 0], dtype='uint8')
        track_file = 'h1.bmp'
        sections = None

    elif track_name == 'PALYA3':
        # palya 3-hoz
        sections = np.array([[350,  60, 350, 100],
                        [425, 105, 430, 95],
                        [500, 140, 530, 110],
                        [520, 160, 580, 150]])

        startline = np.array([ 35, 200,  70, 200])
        endline = np.array([250,  60, 250, 100])
        trk_col = np.array([99, 99, 99])  # pálya színe (szürke)
        track_inside_color = np.array([255, 0, 0], dtype='uint8')
        track_file = 'PALYA3.bmp'

    elif track_name == 'palya4':
        # palya4 teljes
        startline = np.array([273, 125, 273, 64])
        endline = np.array([100, 250, 180, 250])
        trk_col = np.array([99, 99, 99])  # pálya színe (szürke)
        track_inside_color = np.array([255, 0, 0], dtype='uint8')
        track_file = 'PALYA4.bmp'
        sections = None

    elif  track_name == 'palya5':
        # palya5.bmp-hez:
        startline = np.array([670, 310, 670, 130])  # [333, 125, 333, 64],[394, 157, 440, 102],
        endline = None
        trk_col = np.array([99, 99, 99])  # pálya színe (szürke)
        track_inside_color = np.array([255, 0, 0], dtype='uint8')
        track_file = 'PALYA5.bmp'
        sections = None

    else:
        raise ValueError('With this name no track is found')

    return track_file, trk_col, track_inside_color, startline, endline, sections







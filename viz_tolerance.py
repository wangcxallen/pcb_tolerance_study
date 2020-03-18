# -*- coding: utf-8 -*-
"""
Visualization of Tolerance Model
Main function

@author: Wang Chenxi
"""

# Libraries
import numpy as np
import tolerance_module as ts

if __name__ == '__main__':   
    # Configuration Unit:mm
    '''
    Notes:
        1. Don't understand MT-1-202-A101-M200-RS
        2. Which one we are gonna use 50V 100uF 8X12
        3. Don't understand XH-4AW
        4. TSA343G00-250J4 dosen't have pins
        5. No pins' diameter data, HX25003-5AWB
        6. Don't quite understand SR-1-102-C03-0003-RS
        7. Don't understand U217-041N-4BV81
        8. DC-044A-2.5A-2.5 has square pins
        9. DB-9P is really fucking special
        10. RB1-ZZ-0266 also has square pins
        11. PJ-328B0-SMT has really weird shapes, square bent pins
        12. 1-2199298-2 could be approximated
    '''
    com = []
    
    # 0
    com.append({'name':'8x1-Socket',
            'x':np.array([2.5*i for i in np.arange(8)]),
            'y':np.zeros(8),
            'dx':np.zeros(8),
            'dy':np.zeros(8),
            'Rh':0.495,
            'Rp':0.25})# Rp must less than Rh
    # 1
    com.append({'name':'WJ142R-5.08-3P',
            'x':np.array([0.0,5.08,5.08*2,0.0,5.08,5.08*2]),
            'y':np.array([0.0,0.0,0.0,7.62,7.62,7.62]),
            'dx':np.zeros(6),
            'dy':np.zeros(6),
            'Rh':0.7,
            'Rp':0.5})# Rp must less than Rh
    # 2
    com.append({'name':'Z-230010810209',
            'x':np.array([2.54*i for i in np.arange(15)]+[2.54*i for i in np.arange(15)]),
            'y':np.array([0.0]*15 + [2.54]*15),
            'Rh':0.32,
            'Rp':0.5})# Rp must less than Rh
    # 3
    # 2-24pin?
    com.append( {'name':'KF2EDGR-2.5-2P',
            'x':np.array([2.54*i for i in np.arange(8)]),
            'y':np.array([0]*8),
            'Rh':3.5, # Not sure
            'Rp':0.7} )# Rp must less than Rh
    # 4
    com.append( {'name':'1-2199298-2',
            'x':np.array([2.54*i for i in np.arange(8)]),
            'y':np.array([0]*8),
            'Rh':0.5, # Not sure
            'Rp':0.3} )# Rp must less than Rh
    # 5
    com.append( {'name':'MT-1',
            'x':np.array([4.7*i for i in np.arange(3)]*2),
            'y':np.array([0]*3 + [4.8]*3),
            'Rh':1.2,
            'Rp':1.077} )# This is approximated. Rp must less than Rh
    # 6
    com.append( {'name':'SR-1',
            'x':np.array([2.54*i for i in np.arange(3)] + [0.0, 5.08]),
            'y':np.array([0.0]*3 + [5.1]*2),
            'Rh':0.55,
            'Rp':0.49} )# This is approximated. Rp must less than Rh
    # 7
    com.append( {'name':'RB-1-ZZ',
            'x':np.array([0.0, 15.55]),
            'y':np.array([0.0, 0.0]),
            'Rh':0.8,
            'Rp':0.6} )# This is approximated. Rp must less than Rh
    
    for item in com[6:7]:
        insertion = ts.InsertionClass(item['x'], item['y'], item['Rh'], item['Rp'])
        insertion.single_plot(name=item['name'])
        # insertion.double_plot(item['dx'], item['dy'], name=item['name'])
        # insertion.difference_plot(item['dx'], item['dy'], name=item['name'])
        # insertion.center_plot2D(item['dx'], item['dy'])

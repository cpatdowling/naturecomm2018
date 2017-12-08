from datetime import *
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import *
from seaborn import xkcd_rgb as xkcd

#function definitions
def read_delay_file(tfile):
    with open("/home/chase/projects/net-queue/data/congestion/" + tfile, "r") as infile:
        delays = {}
        header = infile.readline()
        lines = infile.readlines()
        delays["dates"] = []
        delays["best"] = []
        delays["avg"] = []
        delays["worst"] = []
        for l in lines:
            line = l.strip().split(",")
            dt = datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S")
            ndt = datetime(year=2016, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute, second=dt.second)
            delays["dates"].append(ndt)
            delays["best"].append(float(line[1]))
            delays["avg"].append(float(line[2]))
            delays["worst"].append(float(line[3]))
    return(delays)
    
def read_traffic_vol(inter, lanes, cards, delay_dates):
    print("reading traffic volume file")
    #inter string, lanes list, cards list
    voldata = {}
    files = os.listdir("/home/chase/projects/net-queue/data/congestion")
    for f in files:
        if f[0:7] == inter:
            if int(f[8]) in cards:
                if int(f[10]) in lanes:
                    with open("/home/chase/projects/net-queue/data/congestion/" + f) as infile:
                        lines = infile.readlines()
                        for line in lines:
                            tokens = line.strip().split(",")
                            dt = datetime.strptime(tokens[0].rstrip(" PDT"), "%m/%d/%Y %H:%M:%S")
                            try:
                                vol = float(tokens[2])
                            except:
                                vol = np.nan
                            if dt not in voldata.keys():
                                voldata[dt] = [vol]
                            else:
                                voldata[dt].append(vol)

    voldates = sorted(voldata.keys())
    for dt in voldates:
        voldata[dt] = np.nansum(np.array(voldata[dt]))

    for dt in delay_dates:
        if dt not in voldata.keys():
            voldata[dt] = np.nan
            
    return(voldata)
    
def quarter_delay_by_status(inter, card, status, delay_dates, out_delays, delay_voldata):
    outputs = {}
    outputs["delay"] = []
    outputs["vol"] = []
    
    avgvols = {}
    for dt in delay_dates:
        if dt.weekday() < 7: #currently excluding sunday
            h = dt.hour
            m = dt.minute
            if h not in avgvols.keys():
                avgvols[h] = {}
            if m not in avgvols[h].keys():
                avgvols[h][m] = {}
                avgvols[h][m]["delay"] = []
                avgvols[h][m]["vol"] = []
            dateindex = out_delays["dates"].index(dt)
            avgvols[h][m]["delay"].append(out_delays[status][dateindex])
            avgvols[h][m]["vol"].append(delay_voldata[dt]/2.0) #averaging cardinalities

    hrs = sorted(avgvols.keys())
    for h in hrs:
        mins = sorted(avgvols[h].keys())
        for m in mins:
            outputs["delay"].append(np.nanmean(avgvols[h][m]["delay"]))
            outputs["vol"].append(np.nanmean(avgvols[h][m]["vol"]))
            
    return(outputs)
    
def linear_fit(volarray, delayarray):
    x = np.array(volarray)
    y = np.array(delayarray)

    A = np.vstack([x, np.ones(len(x))]).T
    res = np.linalg.lstsq(A, y)
    return(res)

def linear_model(fit, volvalue):
    m, c = fit[0]
    delay = m*volvalue + c
    return(delay)
    
def get_delay_by_status(volume, status, fits_by_status):
    fit = fits_by_status[status]
    delay = linear_model(fit, volume)
    return(delay)
    
def get_blocks_parking_traffic(day, cardinal, lanes):
    first_north = [1018, 46254, 1022, 24042, 1026, 24046, 68922, 1030, 1034]
    first_south = [1017,46253, 1021, 24041, 1025, 24045, 68921, 1029, 1033]
    first = first_north + first_south

    #now I need expected rejections by hour on all block-faces adjacent to 1st and 2nd avenues
    with open("../data/ekeytolatlong_2016Q2.pck", 'r') as pick:
        elementkeytolatlong = pickle.load(pick)

    blockids = sorted(elementkeytolatlong)

    block_feeders = []
    block_outs = []
    
    blocks = []
    if 4 in cardinal:
        blocks += first_south
    if 8 in cardinal:
        blocks += first_north

    with open("../data/belltown-blockface-accessibility.csv") as f:
        header = f.readline()
        data = f.readlines()
        for line in data:
            tokens = line.split("#")
            ekeys = tokens[0].strip().split(",")
            origin = ekeys.pop(0).strip()
            if len(origin) > 0:
                origin = int(origin)
                ekeys = [ int(ek.strip()) for ek in ekeys if len(ek) > 1 ]
                if origin in blocks:
                    block_outs += ekeys
                for ek in ekeys:
                    if ek in blocks:
                        block_feeders.append(origin)

    blocks = list(set(blocks + block_feeders))
    blocks = list(set(blocks))
    blocks_i = [ blockids.index(bid) for bid in blocks ]
    return(blocks_i, block_feeders)
    
def get_hourly_vol_data(voldata, delay_dates):
    hourly_voldata = {}
    for i in range(len(delay_dates)/4):
        dt = delay_dates[4*i]
        vol = 0
        for j in range(4):
            dtiter = delay_dates[4*i + j]
            vol += voldata[dtiter]
        if dt.weekday() not in hourly_voldata.keys():
            hourly_voldata[dt.weekday()] = {}
        if dt.hour not in hourly_voldata[dt.weekday()].keys():
            hourly_voldata[dt.weekday()][dt.hour] = [vol]
        else:
            hourly_voldata[dt.weekday()][dt.hour].append(vol)

    weekdays = sorted(hourly_voldata.keys())
    for wd in weekdays:
        hours = hourly_voldata[wd].keys()
        for h in hours:
            hourly_voldata[wd][h] = np.nanmean(np.array(hourly_voldata[wd][h]))
    return(hourly_voldata)        


def get_daily_blocks_traffic(hourly_voldata, day):
    week = ["Monday","Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    weekdays = [0, 1, 2, 3, 4, 5, 6]
    traff_out = []
    weekday = week.index(day)
    traffic_vol_dict = hourly_voldata[weekday]
    hours = sorted(traffic_vol_dict.keys())
    for h in hours:
        traff_out.append(traffic_vol_dict[h])
    return(traff_out)

def get_daily_blocks_rejections(blocklist_i, day):
    rejection_data = 60.0/np.loadtxt("../data/rej_rates_localuniform_fixservice_2016Q2/" + day + "_blockface_Rejections_By_Block.csv", delimiter=",")
    hourly_total = np.zeros((rejection_data.shape[1],))
    for i in blocklist_i:
        hourly_total += rejection_data[i,:]
    return(list(hourly_total))
    
def worst_percent_diffs(parking_traff, reg_traff, fits_by_status):
    diff_traff = [ reg_traff[i] - parking_traff[i] for i in range(len(reg_traff))]
    worst_delay = [ get_delay_by_status(float(v)/4.0, "worst", fits_by_status) for v in reg_traff ]
    
    worst_no_parking_delay = [ get_delay_by_status(float(v)/4.0, "worst", fits_by_status) for v in diff_traff ]
    worst_p_increase = [ (1  - worst_no_parking_delay[i]/worst_delay[i]) for i in range(len(diff_traff)) ]
    return(worst_p_increase)
    
def worst_time_delays(parking_traff, reg_traff, fits_by_status):
    diff_traff = [ reg_traff[i] - parking_traff[i] for i in range(len(reg_traff))]
    worst_delay = [ get_delay_by_status(float(v)/4.0, "worst", fits_by_status) for v in reg_traff ]
    
    worst_no_parking_delay = [ get_delay_by_status(float(v)/4.0, "worst", fits_by_status) for v in diff_traff ]
    worst_p_increase = [ (1  - worst_no_parking_delay[i]/worst_delay[i]) for i in range(len(diff_traff)) ]
    return(worst_delay, worst_no_parking_delay)

    
    
#some stackoverflow plotting functions    
def on_draw(event):
    """Auto-wraps all text objects in a figure at draw-time"""
    import matplotlib as mpl
    fig = event.canvas.figure

    # Cycle through all artists in all the axes in the figure
    for ax in fig.axes:
        for artist in ax.get_children():
            # If it's a text artist, wrap it...
            if isinstance(artist, mpl.text.Text):
                autowrap_text(artist, event.renderer)

    # Temporarily disconnect any callbacks to the draw event...
    # (To avoid recursion)
    func_handles = fig.canvas.callbacks.callbacks[event.name]
    fig.canvas.callbacks.callbacks[event.name] = {}
    # Re-draw the figure..
    fig.canvas.draw()
    # Reset the draw event callbacks
    fig.canvas.callbacks.callbacks[event.name] = func_handles

def autowrap_text(textobj, renderer):
    """Wraps the given matplotlib text object so that it exceed the boundaries
    of the axis it is plotted in."""
    import textwrap
    # Get the starting position of the text in pixels...
    x0, y0 = textobj.get_transform().transform(textobj.get_position())
    # Get the extents of the current axis in pixels...
    clip = textobj.get_axes().get_window_extent()
    # Set the text to rotate about the left edge (doesn't make sense otherwise)
    textobj.set_rotation_mode('anchor')

    # Get the amount of space in the direction of rotation to the left and 
    # right of x0, y0 (left and right are relative to the rotation, as well)
    rotation = textobj.get_rotation()
    right_space = min_dist_inside((x0, y0), rotation, clip)
    left_space = min_dist_inside((x0, y0), rotation - 180, clip)

    # Use either the left or right distance depending on the horiz alignment.
    alignment = textobj.get_horizontalalignment()
    if alignment is 'left':
        new_width = right_space 
    elif alignment is 'right':
        new_width = left_space
    else:
        new_width = 2 * min(left_space, right_space)

    # Estimate the width of the new size in characters...
    aspect_ratio = 0.5 # This varies with the font!! 
    fontsize = textobj.get_size()
    pixels_per_char = aspect_ratio * renderer.points_to_pixels(fontsize)

    # If wrap_width is < 1, just make it 1 character
    wrap_width = max(1, new_width // pixels_per_char)
    try:
        wrapped_text = textwrap.fill(textobj.get_text(), wrap_width)
    except TypeError:
        # This appears to be a single word
        wrapped_text = textobj.get_text()
    textobj.set_text(wrapped_text)

def min_dist_inside(point, rotation, box):
    """Gets the space in a given direction from "point" to the boundaries of
    "box" (where box is an object with x0, y0, x1, & y1 attributes, point is a
    tuple of x,y, and rotation is the angle in degrees)"""
    from math import sin, cos, radians
    x0, y0 = point
    rotation = radians(rotation)
    distances = []
    threshold = 0.0001 
    if cos(rotation) > threshold: 
        # Intersects the right axis
        distances.append((box.x1 - x0) / cos(rotation))
    if cos(rotation) < -threshold: 
        # Intersects the left axis
        distances.append((box.x0 - x0) / cos(rotation))
    if sin(rotation) > threshold: 
        # Intersects the top axis
        distances.append((box.y1 - y0) / sin(rotation))
    if sin(rotation) < -threshold: 
        # Intersects the bottom axis
        distances.append((box.y0 - y0) / sin(rotation))
    return min(distances)


import sys

print("Using python kernel: " + sys.version)
tokens = sys.version.split(".")
if tokens[0] != '2' and tokens[1] != '7':
    print("Warning: this code was compiled and tested with Python 2.7+ on a unix OS\n")
    print("You are using Python version: " + sys.version)
    print("Functions such as xrange won't work in newer versions")
 
#imports
from datetime import *
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import *
from seaborn import xkcd_rgb as xkcd #not neccessary, just delete colors in plotting lines below

#this is a local file containing some data read and plotting functions
sys.path.append("./utils.py")
from utils import *

#globals
intersections = {"apeg263": "1st and Lenora"}
week = ["Monday","Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
weekdays = [0, 1, 2, 3, 4, 5, 6]

tfiles = [ "1st_broad_to_stewart.csv", 
          "1st_stewart_to_broad.csv"]
tfile_map = { "1st_broad_to_stewart.csv": "apeg263", 
             "1st_stewart_to_broad.csv": "apeg263"}

#4: south-east, 8: north-east
cardinals = {4: "Southbound", 8: "Northbound"}

inter = "apeg263"
card = [4, 8] #[4, 8] use both when calculating congestion curve
lanes = [1, 2] #[1, 2] all lanes

#plot font sizes
fl = 22
fs = 20
fs_ = 22

day = sys.argv[1]

#blockface indecies along first avenue north and southbound
first_north = [1018, 46254, 1022, 24042, 1026, 24046, 68922, 1030, 1034]
first_south = [1017,46253, 1021, 24041, 1025, 24045, 68921, 1029, 1033]
first = first_north + first_south


#main script
out_delays = {}
out_delays["dates"] = []
out_delays["best"] = []
out_delays["avg"] = []
out_delays["worst"] = []
for tfile in tfiles:
    delays = read_delay_file(tfile)
    delay_dates = sorted(delays["dates"])
    #check is sorted
    print("Sanity check dates are sorted: ")
    print(all(delay_dates[i] <= delay_dates[i+1] for i in xrange(len(delay_dates)-1)))
    out_delays["dates"] = delay_dates
    out_delays["best"].append(delays["best"])
    out_delays["avg"].append(delays["avg"])
    out_delays["worst"].append(delays["worst"])

types = ["best", "avg", "worst"]
for t in types:
    out_delays[t] = list(1/float(len(tfiles)) * np.nansum(np.asarray(out_delays[t]), axis=0))
    
delay_voldata = read_traffic_vol(inter, lanes, card, delay_dates)
voldata_north = read_traffic_vol(inter, lanes, [8], delay_dates)
voldata_south = read_traffic_vol(inter, lanes, [4], delay_dates)

#delay status
best = quarter_delay_by_status(inter, card, "best", delay_dates, out_delays, delay_voldata)
avg = quarter_delay_by_status(inter, card, "avg", delay_dates, out_delays, delay_voldata)
worst = quarter_delay_by_status(inter, card, "worst", delay_dates, out_delays, delay_voldata)

fits_by_status = {}
for status in types:
    data = quarter_delay_by_status(inter, card, status, delay_dates, out_delays, delay_voldata)
    fit = linear_fit(data["vol"], data["delay"])
    fits_by_status[status] = fit
    
xran = 125
xvals = [ i for i in range(xran) ]
best_fit = [ get_delay_by_status(i, "best", fits_by_status)/60.0 for i in range(xran) ]
worst_fit = [ get_delay_by_status(i, "worst", fits_by_status)/60.0 for i in range(xran) ]
avg_fit = [ get_delay_by_status(i, "avg", fits_by_status)/60.0 for i in range(xran) ]

#delay models
print("plotting delay models in ../figures/congestion_curves.png")
ax = plt.axes()
ax.grid(color=xkcd["grey"], linestyle='dotted')
plt.scatter(best["vol"], [i/60.0 for i in avg["delay"]], color="#67a9cf", s=10)
plt.scatter(worst["vol"], [i/60.0 for i in worst["delay"]], color="#ef8a62", s=10)
plt.scatter(avg["vol"], [i/60.0 for i in best["delay"]], color="#484538", s=10)
plt.plot(xvals, avg_fit, color="#67a9cf", linewidth=3, label="Average")
plt.plot(xvals, worst_fit, color="#ef8a62", linewidth=3, label="Worst")
plt.plot(xvals, best_fit, color="#484538", linewidth=3, label="Best")
plt.ylabel("Travel time in minutes", fontsize=fl)
plt.xlabel("Average volume of vehicles per 15 minutes", fontsize=fl)
plt.setp(ax.get_xticklabels(), fontsize=fs)
plt.setp(ax.get_yticklabels(), fontsize=fs)
plt.xlim(0,xran)
plt.ylim(2.0, 7)
lgd = plt.legend()
plt.legend(fontsize=fl, loc=2)
plt.savefig('../figures/congestion_curves.png',dpi=300, bbox_inches='tight', bbox_extra_artists=(lgd,))
plt.clf()


#traffic congestion results
blocks_i_north, block_feeders_north = get_blocks_parking_traffic(day, [8], lanes)
blocks_i_south, block_feeders_south = get_blocks_parking_traffic(day, [4], lanes)

hourly_voldata_north = get_hourly_vol_data(voldata_north, delay_dates)
hourly_voldata_south = get_hourly_vol_data(voldata_south, delay_dates)

parking_traff_north = [0 for i in range(8)] + get_daily_blocks_rejections(blocks_i_north, day) + [0 for i in range(6)]
parking_traff_south = [0 for i in range(8)] + get_daily_blocks_rejections(blocks_i_south, day) + [0 for i in range(6)]

reg_traff_north = get_daily_blocks_traffic(hourly_voldata_north, day)
reg_traff_south = get_daily_blocks_traffic(hourly_voldata_south, day)

diff_traff_north = np.array(reg_traff_north) - np.array(parking_traff_north)
diff_traff_south = np.array(reg_traff_south) - np.array(parking_traff_south)

prop_north = [ parking_traff_north[i]/reg_traff_north[i] for i in range(len(reg_traff_north)) ]
prop_south = [ parking_traff_south[i]/reg_traff_south[i] for i in range(len(reg_traff_south)) ]

wpdiff_delay_north = worst_percent_diffs(parking_traff_north, reg_traff_north, fits_by_status)
wpdiff_delay_south = worst_percent_diffs(parking_traff_south, reg_traff_south, fits_by_status)

worst_delay_north, worst_no_parking_delay_north = worst_time_delays(parking_traff_north, reg_traff_north, fits_by_status)
worst_delay_south, worst_no_parking_delay_south = worst_time_delays(parking_traff_south, reg_traff_south, fits_by_status)

for c in card:
    if c == 4:
        reg_traff = reg_traff_south
        parking_traff = parking_traff_south
        c_str = "Southbound"
    else:
        reg_traff = reg_traff_north
        parking_traff = parking_traff_north
        c_str = "Northbound"

    hours = []

    firsthour = 6
    lasthour = 22
    for k in range(24):
        if k < 10:
            hours.append("0" + str(k) + ":00")
        else:
            hours.append(str(k) + ":00")
    x = range(24)

    ax = plt.axes()
    ax.grid(color = 'gray', linestyle='dotted')
    plt.plot(x[firsthour:lasthour], reg_traff[firsthour:lasthour], '--', linewidth=3, 
             color=xkcd['yellow orange'], label='Total Traffic  ')
    plt.scatter(x[firsthour:lasthour], reg_traff[firsthour:lasthour], color=xkcd['yellow orange'], s=100)
    plt.plot(x[firsthour:lasthour], parking_traff[firsthour:lasthour], '--', linewidth=3, 
             color='purple', label='Drivers Parking')
    plt.scatter(x[firsthour:lasthour], parking_traff[firsthour:lasthour], color='purple', s=100)
    plt.setp(ax.get_xticklabels(), fontsize=fs)
    plt.setp(ax.get_yticklabels(), fontsize=fs)
    plt.xlim(firsthour,lasthour-1)
    plt.xlabel("Time of day", fontsize=fs_)
    plt.ylabel("Vehicles per hour", fontsize=fs_)
    plt.xticks(range(firsthour,lasthour,2), [hours[firsthour:lasthour][h] for h in range(len(hours[firsthour:lasthour])) if h%2 == 0], rotation=45)
    figure_title = "\n" + day + " " + c_str + " Traffic: 1st Ave"
    ax.set_title(figure_title, fontsize=fs_, position=(0.5,1.1))
    lgd = plt.legend()
    plt.legend()
    plt.legend(fontsize=15,framealpha=100.0, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, borderaxespad=0.)
    print("plotting first avenue traffic and cruising volumes for day " + day + " and cardinality " + c_str + " in ../figures")
    plt.savefig("../figures/" + c_str + '_traffic_' + day + '_first_ave.png',dpi=300, bbox_inches='tight', bbox_extra_artists=(lgd,))
    plt.clf()

for c in card:
    if c == 4:
        reg_traff = reg_traff_south
        c_str = "Southbound"
        clr = xkcd["salmon"]
        linesty = "solid"
    else:
        reg_traff = reg_traff_north
        c_str = "Northbound"
        clr = xkcd["moss"]
        linesty = "solid"

    hours = []

    firsthour = 6
    lasthour = 22
    for k in range(24):
        if k < 10:
            hours.append("0" + str(k) + ":00")
        else:
            hours.append(str(k) + ":00")
    x = range(24)

    ax = plt.axes()
    ax.grid(color = 'gray', linestyle='dotted')
    plt.plot(x[firsthour:lasthour], reg_traff[firsthour:lasthour], '--', linewidth=3.75, color=clr, label=c_str)
    plt.scatter(x[firsthour:lasthour], reg_traff[firsthour:lasthour], color=clr, s=100)
    
plt.setp(ax.get_xticklabels(), fontsize=fs)
plt.setp(ax.get_yticklabels(), fontsize=fs)
plt.xlim(firsthour,lasthour-1)
plt.xlabel("Time of day", fontsize=fs_)
plt.ylabel("Average vehicles per hour", fontsize=fs_)
plt.xticks(range(firsthour,lasthour,2), [hours[firsthour:lasthour][h] for h in range(len(hours[firsthour:lasthour])) if h%2 == 0], rotation=45)
figure_title = "\n" + day + " Traffic: 1st Ave"
ax.set_title(figure_title, fontsize=fs_, position=(0.5,1.1))
lgd = plt.legend()
plt.legend()
plt.legend(fontsize=fs-2,framealpha=100.0, bbox_to_anchor=(0.05, 1.0, 1.0, .102), loc=3,
           ncol=2, borderaxespad=0.)
print("plotting average through traffic for " + day + " in ../figures")
plt.savefig('../figures/average_through_traffic_' + day + '_first_ave.png',dpi=300, bbox_inches='tight', bbox_extra_artists=(lgd,))
plt.clf()

#percent delay increase plot
ax = plt.axes()
ax.grid(color=xkcd["grey"], linestyle='dotted')

cds = [4, 8]
for c in cds:
    if c == 4:
        worst_p_increase = wpdiff_delay_south
        c_str = "Southbound"
        clr = xkcd["salmon"]
        linesty = "solid"
    else:
        worst_p_increase = wpdiff_delay_north
        c_str = "Northbound"
        clr = xkcd["moss"]
        linesty = "solid"
    plt.plot(x[firsthour:lasthour], 100.0*np.array(worst_p_increase)[firsthour:lasthour], '--', linewidth=3.75, color=clr, label=c_str)
    plt.scatter(x[firsthour:lasthour], 100.0*np.array(worst_p_increase)[firsthour:lasthour], color=clr, s=100)
plt.setp(ax.get_xticklabels(), fontsize=fs)
plt.setp(ax.get_yticklabels(), fontsize=fs)
plt.ylim(0,30)
plt.xlim(firsthour,lasthour-1)
plt.xlabel("Time of day: typical " + day, fontsize=fs_)
plt.ylabel("Percent delay \n increase", fontsize=fs_)
plt.xticks(range(firsthour,lasthour,2), 
               [hours[firsthour:lasthour][h] for h in range(len(hours[firsthour:lasthour])) if h%2 == 0], 
               rotation=60)
lgd = plt.legend()
plt.legend(fontsize=fs,framealpha=100.0, loc=2)
plt.tight_layout()
print("plotting marginal time costs per vehicle due to cruising for " + day)
plt.savefig('../figures/marginalcost_' + day + '.png',dpi=300, bbox_inches='tight',bbox_extra_artists=(lgd,))
plt.clf()


#time cost plots
ax = plt.axes()
ax.grid(color=xkcd["grey"], linestyle='dotted')

for c in cds:
    if c == 4:
        worst_delay = worst_delay_south
        worst_no_parking_delay = worst_no_parking_delay_south
        c_str = "SB"
        clr = xkcd["salmon"]
        linesty = "solid"
    else:
        worst_delay = worst_delay_north
        worst_no_parking_delay = worst_no_parking_delay_north
        c_str = "NB"
        clr = xkcd["moss"]
        linesty = "solid"
    plt.plot(x[firsthour:lasthour], (1.0/60.0)*np.array(worst_delay)[firsthour:lasthour], 'b', linewidth=3.75, color=clr, ls=linesty, label=c_str + " worst delay")
    plt.plot(x[firsthour:lasthour], (1.0/60.0)*np.array(worst_no_parking_delay)[firsthour:lasthour], 'b', linewidth=3.75, color=clr, ls="dashed", label=c_str + " without cruising")

plt.setp(ax.get_xticklabels(), fontsize=fs)
plt.setp(ax.get_yticklabels(), fontsize=fs)
plt.xlim(firsthour,lasthour-1)
plt.xlabel("Time of day: typical " + day, fontsize=fs_)
plt.ylabel("Per vehicle delay (minutes)", fontsize=fs_)
plt.xticks(range(firsthour,lasthour,2), 
               [hours[firsthour:lasthour][h] for h in range(len(hours[firsthour:lasthour])) if h%2 == 0], 
               rotation=60)

lgd = plt.legend()
plt.legend()
plt.legend(fontsize=fs-2,framealpha=100.0, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, borderaxespad=0.)
print("plotting total time costs for " + day)
plt.savefig('../figures/timecost_' + day + '.png',dpi=300, bbox_inches='tight',bbox_extra_artists=(lgd,))
plt.clf()
    
    

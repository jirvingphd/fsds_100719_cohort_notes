# import matplotlib.dates as mdates
# import datetime

# # Instantiate Locators to be used
# years = mdates.YearLocator()   # every year
# months = mdates.MonthLocator()#interval=2)  # every month
# quarters = mdates.MonthLocator(interval=3)#interval=2)  # every month

# # Define various date formatting to be used
# monthsFmt = mdates.DateFormatter('%Y-%b')
# yearsFmt = mdates.DateFormatter('%Y') #'%Y')
# yr_mo_day_fmt = mdates.DateFormatter('%Y-%m')
# monthDayFmt = mdates.DateFormatter('%m-%d-%y')


# ## AX2 SET TICK LOCATIONS AND FORMATTING

# # Set locators (since using for both location and formatter)
# auto_major_loc = mdates.AutoDateLocator(minticks=5)
# auto_minor_loc = mdates.AutoDateLocator(minticks=10)

# # Set Major X Axis Ticks
# ax1.xaxis.set_major_locator(auto_major_loc)
# ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(auto_major_loc))

# # Set Minor X Axis Ticks
# ax1.xaxis.set_minor_locator(auto_minor_loc)
# ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(auto_minor_loc))


# ax1.tick_params(axis='x',which='both',rotation=30)
# ax1.grid(axis='x',which='major')
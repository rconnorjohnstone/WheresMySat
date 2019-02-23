import RRE_measurements as RRE

range, range_rate, elev = 300, 1.2, 40
latitude, longitude, height, time = 20,30,58537,7000
station1 = RRE.station(latitude, longitude, height)
range1 = RRE.RangeMeas(range, time, 'New Horizons', station1)
range_rate1 = RRE.RangeRateMeas(range_rate, time, 'New Horizons', station1)
elev1 = RRE.ElevMeas(elev, time, 'New Horizons', station1)

print(station1)
print(range1)
print(range_rate1)
print(elev1)
import RRE_measurements as RRE
import numpy as np

mission1, range1, range_rate1, elev1, sigma1 = "ISS",150, 1.2, 40, 0.01
mission2, range2, range_rate2, elev2, sigma2 = "Soviet Spy Satellite",200, 1.5, -20, 0.04

station1 = RRE.station("Boulder", 40.0, -105.2, 5400)
station2 = RRE.station("Louisville", 38.25, -85.75, 460)

t1, t2 = 58537.863889, 41988.1875

range_meas1 = RRE.RangeMeas(range1, t1, mission1, station1)
range_rate_meas1 = RRE.RangeRateMeas(range_rate1, t1, mission1, station1)
elev_meas1 = RRE.ElevMeas(elev1, t1, mission1, station1)

range_meas2 = RRE.RangeMeas(range2, t1, mission2, station1)
range_rate_meas2 = RRE.RangeRateMeas(range_rate2, t1, mission2, station1)
elev_meas2 = RRE.ElevMeas(elev2, t1, mission2, station1)

range_meas3 = RRE.RangeMeas(range1, t2, mission1, station2)
range_rate_meas3 = RRE.RangeRateMeas(range_rate1, t2, mission1, station2)
elev_meas3 = RRE.ElevMeas(elev1, t2, mission1, station2)

state1 = np.array([8000,1000,2000,1,6.5,-1])
state2 = np.array([6000,3000,-2000,1.9,6.5,-1.2])

range_resid1 = range_meas1.residual(state1)
range_rate_resid1 = range_rate_meas1.residual(state1)
elev_resid1 = elev_meas1.residual(state1)

range_resid2 = range_meas3.residual(state2)
range_rate_resid2 = range_rate_meas3.residual(state2)
elev_resid2 = elev_meas3.residual(state2)

print(station1)
print(station1.get_ECI(t1))
print('')
print(station2)
print(station2.get_ECI(t2))
print('')

print(range_meas1)
print(range_rate_meas1)
print(elev_meas1)
print('')
print(range_meas2)
print(range_rate_meas2)
print(elev_meas2)
print('')
print(range_meas3)
print(range_rate_meas3)
print(elev_meas3)
print('')

print(range_resid1)
print(range_rate_resid1)
print(elev_resid1)
print('')
print(range_resid2)
print(range_rate_resid2)
print(elev_resid2)

# rescales TOA axis for deterministic CW signals
cw_renorm = 1e-10

# reference time for CW model
tref = 4579200000.  # 53000 * day = 53000 * 86400


# times and frequencies
day = 86400.0   # seconds
year = 365.2526 * day   # seconds
fyr = 1 / year    # Hertz
year_months = 12.
year_days = 365.25
us_sec = 1.e-6

# reference time for CW model
tref = 4579200000.  # 53000 * day = 53000 * 86400

# physical constants
c = 299792458.0
G = 6.6743e-11
Msun = 1.9891e30
Tsun = Msun * G / c**3.
kpc = 3.085677581491367e+19
Mpc = 1.e3 * kpc
Tkpc = kpc / c

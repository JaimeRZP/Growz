# Growz
The code utilises PyMC3 to fit Gaussian process regression to a varaity of cosmological datasets and forecasts. 

The implemented data sets so far are:

+Cosmic Chronometers

+Pantheon DS17

+BOSS DR12

+eBOSS DR16

+Wigglez 

+SN DSS

+DESI (forecast)

+WFIRST (forecast)

The code can simultaneously employ data on the Hubble rate, cosmological distances and the growth rate constrain a single Gaussian process on the Hubble rate by virtue of the phyiscal relationships between the three quantities.

The code only makes of the following assumptions:

+The FLRW metric

+The collapse of matter occurs quasi-statically

being otherwise assumption free.

To set your own run go to either scripts/notebooks and edit the master .py/.ipynb file to include your data combination.
You may require the following changes:

+master file: change l12 to '/your_directory/Growz/data/products'.

+make_data.py: change l122 and l126 to '/your_directory/Growz/data/raw/...'

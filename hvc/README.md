This script runs HVC (horizontal vertical clustering) for various epsilon configurations.
We provide a subset of the complete dataset - reviews from December 2022.
Some epsilon configurations recreate some of the results from the paper.

For example, the SelectorsHub & SelectorsHub Pro cluster from Table 5 is generated by the dbscan_hori_0.0001_verti_0.0001 as ('ndgimibanhlabgdgjcpbbndiehljcpfh', 'kodoloplfbnhlfcepehlafnbojbfgglb').

With params dbscan_hori_1e-05_verti_0.0001, we also see some of the spam results detected by this method as well (Table 8).
('loinekcabhlmhjjbocijdoimmejangoa', 'ebfidpplhabeedpnhjnobghokpiioolj'), ('jnldfbidonfeldmalbflbmlebbipcnle', 'loinekcabhlmhjjbocijdoimmejangoa')

To run it:
python3 hvc.py

Some Python packages are needed: tqdm, pandas, seaborn, numpy, matplotlib
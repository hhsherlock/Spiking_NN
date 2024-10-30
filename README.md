## Spiking Neural Network

- 22.10.2024: Initial Commit
    yay

- 30.10.2024: Added ligand gated receptors like AMPA
    I chose random constants for the channel properties, the current goes instantly overflow. 

**HH_original.py**: the original code from https://github.com/swharden/pyHH.

**HH.py**: the new code I wrote with more structured classes and bug fixed. Except for the original Hodgkin Huxley model, I also added ligand gated channels like AMPA, GABA... NMDA behaves differently.

**network.py**: ignore it as well, it is empty. I plan to add different kinds of neurons with different combinations with channels and receptors and connect them. 

**run_original.ipynb**: HH_original classes run. 

**run.ipynb**: HH classes run. 

**test.ipynb**: ignore it, i use it for testing.
